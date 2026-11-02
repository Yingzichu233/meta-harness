"""Similarity-Cluster Hierarchy Memory.

Mechanism (new relative to flat demo banks): memory is a *two-level retrieval
tree over the answer space*, built without any LLM call.

  - Leaves are distinct answers (exact strings); each leaf carries up to two
    exemplar snippets.
  - Learning clusters the leaves with average-linkage agglomeration over a
    lexical similarity on answer *surface* (token + char n-gram + CJK n-gram
    cosine).  The similarity threshold is chosen at render time so that the
    number of groups stays inside a target band ([8, 14] groups), i.e. the tree
    adapts its breadth to how many distinct answers exist instead of using a
    fixed cluster count.
  - Prediction renders the tree top-down: each GROUP is introduced by its most
    central member (the "hub" that the query should be routed to), then its
    members are listed with one-line exemplars.

So the prompt is a routing tree: the solver first decides which group the
problem belongs to, then which member inside the group.  Flat banks force a
single flat similarity competition between hundreds of candidates; the
hierarchy makes the first decision cheap.

Rationale: grouping lets a large answer set be shown in full (no answer is
dropped for lack of context), which should help label-rich datasets, while on
small answer sets the tree degenerates into a plain grouped listing and costs
about the same as a demo bank.
"""

import json
import re
from typing import Any

from ..llm import LLMCallable
from ..memory_system import MemorySystem, extract_json_field

PROMPT_TEMPLATE = """Classify the problem below. The answer space is organised as GROUPS of
related answers; each group names its most common label and lists its members with
one illustrative solved example each.

{hierarchy}

**Problem:**
{input}

**Instructions:**
- First choose the group whose examples resemble the problem
- Then choose the exact member label inside that group that fits best
- Copy the label exactly as written in the group listing
- Respond in JSON format

{{"reasoning": "[brief reasoning]", "final_answer": "[your answer]"}}"""

TARGET_MIN_GROUPS = 8
TARGET_MAX_GROUPS = 14
CTX_BUDGET = 22000
SNIPPET = 130
PER_LABEL_EXAMPLES = 1
MAX_MEMBERS_SHOWN = 16

_CJK = re.compile(r"[\u4e00-\u9fff]")
_STOP = set("the a an of and or is are to in on for with that this it as at by".split())


def _tokens(text: str) -> set[str]:
    low = str(text).lower()
    out = set(w for w in re.findall(r"[a-z0-9]+", low) if w not in _STOP)
    for n in (4, 5):
        for i in range(len(low) - n + 1):
            out.add(low[i : i + n])
    cj = _CJK.findall(low)
    for n in (2, 3):
        for i in range(len(cj) - n + 1):
            out.add("".join(cj[i : i + n]))
    return out


def _cos(a: set[str], b: set[str]) -> float:
    inter = a & b
    if not inter:
        return 0.0
    return len(inter) / ((len(a) * len(b)) ** 0.5 + 1e-9)


def _cluster(labels: list[str], feats: dict[str, set[str]]) -> list[list[str]]:
    """Average-linkage agglomeration until the group count hits the target band."""
    clusters: list[list[str]] = [[l] for l in labels]
    # pre-compute pairwise similarities (label sets are small: <= a few hundred)
    sim: dict[tuple[str, str], float] = {}
    for i, a in enumerate(labels):
        for b in labels[i + 1 :]:
            sim[(a, b)] = sim[(b, a)] = _cos(feats[a], feats[b])

    def avg_sim(c1: list[str], c2: list[str]) -> float:
        tot, n = 0.0, 0
        for a in c1:
            for b in c2:
                tot += sim[(a, b)]
                n += 1
        return tot / n if n else 0.0

    while len(clusters) > TARGET_MAX_GROUPS:
        best, bi, bj = -1.0, -1, -1
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                s = avg_sim(clusters[i], clusters[j])
                if s > best:
                    best, bi, bj = s, i, j
        if bi < 0:
            break
        clusters[bi] = clusters[bi] + clusters[bj]
        clusters.pop(bj)
    return clusters


class SimilarityClusterHierarchyMemory(MemorySystem):
    """Hierarchical (group -> label -> exemplar) memory over the answer space."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self.by_label: dict[str, list[str]] = {}  # label -> exemplar snippets
        self._cache_key: tuple = ()
        self._cache: str = ""

    # ---------------- learning ----------------
    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        for r in batch_results:
            label = str(r["ground_truth"]).strip()
            if not label:
                continue
            q = " ".join(str(r.get("raw_question") or r["input"]).split())[:SNIPPET]
            self.by_label.setdefault(label, []).append(q)

    # ---------------- hierarchy ----------------
    def _build(self) -> str:
        labels = sorted(self.by_label)
        if not labels:
            return ""
        key = tuple((l, len(self.by_label[l])) for l in labels)
        if key == self._cache_key:
            return self._cache

        feats = {l: _tokens(l) for l in labels}
        groups = _cluster(labels, feats)

        parts: list[str] = []
        total = 0
        groups.sort(key=lambda g: -sum(len(self.by_label[l]) for l in g))
        for members in groups:
            size = sum(len(self.by_label[l]) for l in members)
            if len(members) == 1:
                hub = members[0]
            else:
                hub = max(
                    members,
                    key=lambda m: sum(_cos(feats[m], feats[o]) for o in members),
                )
            lines = [f"GROUP (~{size} solved examples) — most central label: {hub}"]
            for label in members[:MAX_MEMBERS_SHOWN]:
                exs = self.by_label[label][:PER_LABEL_EXAMPLES]
                illus = " | ".join(exs)
                lines.append(f"  * {label} ({len(self.by_label[label])}x) e.g. {illus}")
            block = "\n".join(lines)
            if total + len(block) > CTX_BUDGET and parts:
                break
            parts.append(block)
            total += len(block) + 2
        self._cache_key = key
        self._cache = "ANSWER GROUPS:\n\n" + "\n\n".join(parts) if parts else ""
        return self._cache

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        section = self._build()
        prompt = PROMPT_TEMPLATE.format(hierarchy=section, input=input)
        response = self.call_llm(prompt)
        answer = extract_json_field(response, "final_answer")
        return answer, {
            "full_response": response,
            "num_labels": len(self.by_label),
        }

    def get_context_length(self) -> int:
        return len(self._build())

    # ---------------- state ----------------
    def get_state(self) -> str:
        return json.dumps({"by_label": self.by_label}, indent=2, ensure_ascii=False)

    def set_state(self, state: str) -> None:
        data = json.loads(state)
        self.by_label = data.get("by_label", {}) or {}
        self._cache_key = ()
        self._cache = ""
