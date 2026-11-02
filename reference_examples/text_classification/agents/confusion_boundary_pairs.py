"""Confusion Boundary Pairs (novel axis: the unit of memory is a *boundary
between two adjacent answers*, learned across the whole history, not a single
solved case retrieved by similarity).

Motivation, measured on the stored memories: on the long-tail answer space the
k nearest cases almost never share the query's answer (top-5 distinct answers
in ~94% of queries), and on the near-duplicate task ~88% of queries are
answered by a case in the top 5.  What separates the neighbouring answers is
usually a handful of discriminative words, and the flat bank never states it.
Plain "contrastive pair" memory tried to state it but only paired each case
with the single most similar case with a different answer, ranked pairs by the
query's similarity to their positive member -- which, under a long tail, means
the pairs shown are about unrelated answers.

Mechanism here:
  * After every learning step the memory recomputes a persistent *confusion
    graph* over answers: for every solved case, the nearest cases whose answer
    differs are recorded as its boundary partners (weighted by how often the
    pair is observed and how similar the two answer surfaces are, which is
    what catches confusable wording).
  * At prediction time the query's nearest cases vote for an answer.  The
    boundary with the *runner-up* answer -- the actual decision this query
    faces -- is looked up in the graph, and the prompt is built around that
    single comparison: one verbatim case for each of the two answers, their
    answer strings, and the tokens that occur in one and not the other.
  * The remaining budget is filled with a rarest-first coverage listing, which
    makes the rest of the answer space visible.

So the prompt is organised around a decision, with three parts: the pair that
must be resolved, the exact answer strings, and the tail of the answer space.
"""

import json
import re
from typing import Any

from ..llm import LLMCallable
from ..memory_system import MemorySystem, extract_json_field

PROMPT_TEMPLATE = """Solve the problem below.

{boundary}
{coverage}
**Problem:**
{input}

**Instructions:**
- Compare the problem with the two competing answers above; use the
  distinguishing words to decide which one applies
- Copy the answer wording exactly as it appears above
- Respond in JSON format

{{"reasoning": "[brief reasoning]", "final_answer": "[your answer]"}}"""

BUDGET = 26000
SNIP = 220
MAX_CASES = 120
MAX_DIFF = 8
MAX_CONFUSION = 6


def _ws(t: str) -> str:
    return " ".join(str(t).split())


def _clip(s: str) -> str:
    s = _ws(s)
    if len(s) <= SNIP:
        return s
    return s[: SNIP // 2] + " ... " + s[-SNIP // 2 :]


_CJK = re.compile(r"[\u4e00-\u9fff]")
_STOP = set("the a an of and or is are to in on for with that this it as at by".split())


def _tokens(text: str) -> set[str]:
    low = _ws(text).lower()
    out = {w for w in re.findall(r"[a-z0-9]+", low) if w not in _STOP}
    for n in (4, 5):
        for i in range(len(low) - n + 1):
            out.add(low[i : i + n])
    cj = _CJK.findall(low)
    for n in (2, 3):
        for i in range(len(cj) - n + 1):
            out.add("".join(cj[i : i + n]))
    for n in (2, 3):
        for i in range(len(low) - n + 1):
            s = low[i : i + n]
            if s.isascii() and s.isalnum():
                out.add(s)
    return out


def _sim(a: set[str], b: set[str]) -> float:
    inter = a & b
    if not inter:
        return 0.0
    return len(inter) / ((len(a) * len(b)) ** 0.5 + 1e-9)


class ConfusionBoundaryPairs(MemorySystem):
    """A bank whose prompt is organised around the boundary this query faces."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self.examples: list[dict[str, Any]] = []          # {q, a, tok}
        self.confused: dict[str, dict[str, float]] = {}   # answer -> {answer: weight}
        self._stale = 0                                   # cases added since rebuild

    # ---------------- learning ----------------
    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        for r in batch_results:
            a = _ws(r["ground_truth"])
            if not a:
                continue
            q = _ws(r.get("raw_question") or r["input"])
            self.examples.append({"q": q, "a": a, "tok": frozenset(_tokens(q))})
            self._stale += 1
        if self._stale:
            self._rebuild_confusion()
            self._stale = 0

    def _rebuild_confusion(self) -> None:
        ex = self.examples
        if len(ex) < 2:
            self.confused = {}
            return
        ans_tok: dict[str, set[str]] = {}
        for e in ex:
            ans_tok.setdefault(e["a"], set()).update(_tokens(e["a"]))
        conf: dict[str, dict[str, float]] = {}
        # for every case, its nearest cases whose answer differs
        for i, e in enumerate(ex):
            near = sorted(
                (
                    (_sim(e["tok"], ex[j]["tok"]), ex[j]["a"])
                    for j in range(len(ex))
                    if ex[j]["a"] != e["a"]
                ),
                reverse=True,
            )[:MAX_CONFUSION]
            for s, other in near:
                if s <= 0.0:
                    continue
                w = s + 0.5 * _sim(ans_tok.get(e["a"], set()), ans_tok.get(other, set()))
                row = conf.setdefault(e["a"], {})
                row[other] = row.get(other, 0.0) + w
        self.confused = conf

    # ---------------- rendering ----------------
    def _pick_pair(self, scored: list[int], input_tok: set[str]) -> tuple[int, int] | None:
        """Return (case index for answer A, case index for answer B)."""
        if not scored:
            return None
        best_i = scored[0]
        votes: dict[str, float] = {}
        rep: dict[str, int] = {}
        for rank, i in enumerate(scored[:12]):
            a = self.examples[i]["a"]
            votes[a] = votes.get(a, 0.0) + 1.0 / (1 + rank)
            rep.setdefault(a, i)
        order = sorted(votes, key=lambda a: -votes[a])
        # runner-up: highest log-odds vs the leading answer among the runners
        if len(order) < 2:
            return None
        lead = order[0]
        row = self.confused.get(lead, {})
        runner_cands = [a for a in order[1:]]
        extra = [a for a in sorted(row, key=lambda a: -row[a])[:3] if a not in runner_cands]
        pool = runner_cands + extra
        if not pool:
            return None
        runner = max(pool, key=lambda a: (row.get(a, 0.0), votes.get(a, 0.0)))
        j = rep.get(runner)
        if j is None:  # answer never seen as a near case: use its own representative
            j = next((k for k, e in enumerate(self.examples) if e["a"] == runner), None)
        if j is None:
            return None
        return best_i, j

    def _render(self, input: str) -> tuple[str, str]:
        if not self.examples:
            return "", ""
        qtok = _tokens(input)
        scored = sorted(
            range(len(self.examples)),
            key=lambda i: -_sim(qtok, self.examples[i]["tok"]),
        )
        chosen: list[int] = []
        used = 0
        boundary = ""
        pair = self._pick_pair(scored, qtok)
        if pair is not None:
            i, j = pair
            ea, eb = self.examples[i], self.examples[j]
            ta, tb = _tokens(ea["q"]), _tokens(eb["q"])
            # distinguishing words: prefer word-like tokens, then n-grams
            diff = [t for t in ta - tb if 2 <= len(t) <= 20][: MAX_DIFF * 2]
            diff.sort(key=lambda t: (len(t) > 12, len(t)))
            diff_s = ", ".join(diff[:MAX_DIFF]) if diff else "(no salient difference)"
            boundary = (
                "COMPETING ANSWERS (two very similar solved problems with "
                "different answers)\n"
                f"Case 1\nQ: {_clip(ea['q'])}\nA: {ea['a']}\n"
                f"Case 2\nQ: {_clip(eb['q'])}\nA: {eb['a']}\n"
                f"Words that point to Case 1 and not Case 2: {diff_s}\n\n"
            )
            chosen.extend([i, j])
            used += len(boundary)

        counts: dict[str, int] = {}
        for e in self.examples:
            counts[e["a"]] = counts.get(e["a"], 0) + 1
        first: dict[str, int] = {}
        for i, e in enumerate(self.examples):
            first.setdefault(e["a"], i)
        blocks = []
        for ans in sorted(first, key=lambda k: counts[k]):
            i = first[ans]
            if i in chosen:
                continue
            block = f"Q: {_clip(self.examples[i]['q'])}\nA: {self.examples[i]['a']}"
            if used + len(block) + 2 > BUDGET or len(chosen) >= MAX_CASES:
                continue
            blocks.append(block)
            chosen.append(i)
            used += len(block) + 2
        coverage = ""
        if blocks:
            coverage = (
                "OTHER ANSWER CATEGORIES SEEN (rarest first):\n\n"
                + "\n\n".join(blocks)
                + "\n\n"
            )
        return boundary, coverage

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        boundary, coverage = self._render(input)
        prompt = PROMPT_TEMPLATE.format(
            boundary=boundary, coverage=coverage, input=input
        )
        response = self.call_llm(prompt)
        answer = extract_json_field(response, "final_answer")
        return answer, {
            "full_response": response,
            "has_boundary": bool(boundary),
            "num_examples": len(self.examples),
        }

    def get_context_length(self) -> int:
        if not self.examples:
            return 0
        b, c = self._render(self.examples[len(self.examples) // 2]["q"])
        return len(b) + len(c)

    # ---------------- state ----------------
    def get_state(self) -> str:
        return json.dumps(
            {
                "examples": [
                    {"q": e["q"], "a": e["a"], "tok": sorted(e["tok"])}
                    for e in self.examples
                ],
                "confused": self.confused,
            },
            ensure_ascii=False,
        )

    def set_state(self, state: str) -> None:
        d = json.loads(state)
        self.examples = [
            {"q": e["q"], "a": e["a"], "tok": frozenset(e.get("tok", []))}
            for e in d.get("examples", [])
        ]
        self.confused = d.get("confused", {})
        if self.examples and not self.confused:
            self._rebuild_confusion()
        self._stale = 0
