"""Label Catalogue Memory.

Mechanism (new relative to demo-listing memories): the memory is not a list of
examples, it is a *label catalogue with one representative demo per label*.

  - In learning, every solved example is reduced to its label string and a
    short representative snippet. Only ONE snippet per distinct label is kept
    (the first-seen, capped in length), so memory size scales with the number
    of distinct labels, not with the number of training examples.
  - In prediction, a lexical (token + char n-gram + CJK n-gram, IDF weighted)
    similarity search over the representatives selects the subset of the
    catalogue relevant to the query, and the prompt presents the task as
    "pick exactly one label from this catalogue", listing each candidate label
    once with its representative snippet.

Rationale: raw demo banks spend their whole context budget on duplicate labels
and on examples that are not discriminative; the catalogue keeps one
discriminative prototype per class and a compact label inventory, which is the
information the solver actually needs to commit to a label.
"""

import hashlib
import json
import math
import re
from collections import Counter
from typing import Any

from ..llm import LLMCallable
from ..memory_system import MemorySystem, extract_json_field

PROMPT_TEMPLATE = """You must classify the problem below into exactly one of the candidate answers listed.

{examples_section}

**Problem:**
{input}

**Instructions:**
- Choose the single candidate that best matches the problem
- Copy the candidate answer EXACTLY as written above (casing, punctuation, separators)
- Respond in JSON format

{{"reasoning": "[brief reasoning]", "final_answer": "[your answer]"}}"""

MAX_SNIPPET = 420
MAX_CANDIDATES = 30
CTX_BUDGET = 9000
TOP_K_NEIGHBOURS = 12

_CJK = re.compile(r"[\u4e00-\u9fff]")


def _tokens(text: str) -> set[str]:
    low = text.lower()
    out: set[str] = set(re.findall(r"[a-z0-9]+", low))
    for n in (4, 5):
        for i in range(len(low) - n + 1):
            out.add(low[i : i + n])
    cjk = _CJK.findall(low)
    for n in (2, 3):
        for i in range(len(cjk) - n + 1):
            out.add("".join(cjk[i : i + n]))
    return out


def _seed_for_input(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")


class LabelCatalogueMemory(MemorySystem):
    """Memory of distinct labels with one representative snippet each."""

    def __init__(self, llm: LLMCallable, max_candidates: int = MAX_CANDIDATES):
        super().__init__(llm)
        self.reps: list[dict[str, str]] = []  # {label, snippet}
        self._labels: set[str] = set()
        self.max_candidates = max_candidates

    # ---------------- learning ----------------
    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        for r in batch_results:
            label = str(r["ground_truth"]).strip()
            if not label or label in self._labels:
                continue
            self._labels.add(label)
            snippet = str(r.get("raw_question") or r["input"]).strip()
            snippet = " ".join(snippet.split())[:MAX_SNIPPET]
            self.reps.append({"label": label, "snippet": snippet})

    # ---------------- prediction ----------------
    def _select(self, input: str) -> tuple[list[dict[str, str]], int]:
        if not self.reps:
            return [], 0
        if len(self.reps) <= self.max_candidates:
            # everything fits: show the whole catalogue, ordered deterministically
            chosen = sorted(self.reps, key=lambda r: r["label"])
        else:
            qtok = _tokens(input)
            df = Counter()
            docs = []
            for rep in self.reps:
                toks = _tokens(rep["snippet"] + " " + rep["label"])
                docs.append(toks)
                df.update(toks)
            n = len(docs)
            default_idf = math.log(1 + (n + 0.5) / 0.5)
            scored = []
            for i, toks in enumerate(docs):
                inter = qtok & toks
                if not inter:
                    continue
                s = 0.0
                for t in inter:
                    idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
                    s += idf
                s /= len(qtok) ** 0.5 * (len(toks) ** 0.5 + 1e-9)
                scored.append((s, i))
            scored.sort(reverse=True)
            keep = [i for _, i in scored[:TOP_K_NEIGHBOURS]]
            order = {i: k for k, i in enumerate(keep)}
            chosen = [self.reps[i] for i in sorted(keep, key=lambda i: order[i])]
            # top up with the most recent unseen labels so the catalogue stays broad
            seen = set(keep)
            for i in range(len(self.reps) - 1, -1, -1):
                if len(chosen) >= self.max_candidates:
                    break
                if i not in seen:
                    chosen.append(self.reps[i])

        parts, total = [], 0
        for rep in chosen:
            part = f'Candidate answer: {rep["label"]}\nExample problem: {rep["snippet"]}'
            if total + len(part) > CTX_BUDGET and parts:
                break
            parts.append(part)
            total += len(part) + 2
        return parts and chosen[: len(parts)] or [], total

    def _format(self, input: str) -> str:
        chosen, _ = self._select(input)
        return "\n\n".join(
            f'Candidate answer: {r["label"]}\nExample problem: {r["snippet"]}'
            for r in chosen
        )

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        section = self._format(input)
        prompt = PROMPT_TEMPLATE.format(examples_section=section, input=input)
        response = self.call_llm(prompt)
        answer = extract_json_field(response, "final_answer")
        return answer, {
            "full_response": response,
            "num_labels": len(self.reps),
        }

    def get_context_length(self) -> int:
        if not self.reps:
            return 0
        mid = self.reps[len(self.reps) // 2]["snippet"]
        return len(self._format(mid))

    # ---------------- state ----------------
    def get_state(self) -> str:
        return json.dumps({"reps": self.reps}, indent=2)

    def set_state(self, state: str) -> None:
        data = json.loads(state)
        self.reps = data.get("reps", [])
        self._labels = {r["label"] for r in self.reps}
