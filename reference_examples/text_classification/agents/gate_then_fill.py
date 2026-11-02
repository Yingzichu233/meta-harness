"""Gate-Then-Fill (novel axis: the demo budget is a BUDGET and surplus must be
justified by evidence, instead of being refilled every query).

Two measured facts drive it:
  (a) On a long-tail answer space (143 distinct answers among 200 cases, 107 of
      them singletons) pure nearest-neighbour retrieval returns a *different*
      answer for the top-5 neighbours in ~94% of queries, so any scheme that
      spends its characters on the k nearest cases is spending them on
      answers that each occur exactly once.  On a task whose queries are
      near-duplicates of each other the same top-k is 74-88% correct.
  (b) The two regimes are separable without knowing anything about the task:
      the mean nearest-neighbour self-similarity of the memory is ~0.76, ~0.47
      and ~0.15 in the three regimes, and every query/prior ratio sits near 1.

So the widening path is GATED on evidence rather than always on:
  * Learning appends cases with a token fingerprint and maintains the
    self-similarity prior online.
  * Prediction always fills the *coverage* part of the budget: one
    representative per distinct answer, rarest answer first.  This is the part
    that carries the output space.
  * Only if the query's own nearest-neighbour similarity is high BOTH
    absolutely (>= NEAR_DUP) and relative to the memory's prior (>= RATIO,
    set at 2.0 because the measured per-query ratio clusters near 1.0 for
    every regime, so the gate must sit well outside that band) does the
    system spend one extra slot on that nearest case as a worked precedent.
    The gate closes automatically for tasks where answers are not reusable,
    so the long-tail task receives the pure coverage bank (as the strongest
    baseline does) while a genuinely near-duplicate query still gets the
    nearest case as guidance.

No clustering, no rewriting of answers, no per-dataset constants.
"""

import json
import re
from typing import Any

from ..llm import LLMCallable
from ..memory_system import MemorySystem, extract_json_field

PROMPT_TEMPLATE = """Solve the problem below based on the solved cases provided.

{precedent}{solved_cases}

**Problem:**
{input}

**Instructions:**
- Answer exactly in the form used by the solved cases
- Respond in JSON format

{{"reasoning": "[brief reasoning]", "final_answer": "[your answer]"}}"""

BUDGET = 26000      # characters of injected context
SNIP = 200          # characters per case (head + tail when longer)
NEAR_DUP = 0.60     # absolute level: queries are near-duplicates of solved cases
RATIO = 2.00        # relative level: the query is closer than usual for this task
MAX_CASES = 120

_CJK = re.compile(r"[\u4e00-\u9fff]")
_STOP = set("the a an of and or is are to in on for with that this it as at by".split())


def _tokens(text: str) -> set[str]:
    low = " ".join(str(text).lower().split())
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


def _clip(s: str) -> str:
    s = " ".join(str(s).split())
    if len(s) <= SNIP:
        return s
    return s[: SNIP // 2] + " ... " + s[-SNIP // 2 :]


class GateThenFillMemory(MemorySystem):
    """Coverage bank always; nearest-case precedent only when the gate is open."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self.examples: list[dict[str, Any]] = []   # {q, a, tok, self}

    # ---------------- learning ----------------
    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        for r in batch_results:
            a = " ".join(str(r["ground_truth"]).split())
            if not a:
                continue
            q = " ".join(str(r.get("raw_question") or r["input"]).split())
            tok = frozenset(_tokens(q))
            best = max((_sim(tok, e["tok"]) for e in self.examples), default=0.0)
            self.examples.append({"q": q, "a": a, "tok": tok, "self": best})

    def _prior(self) -> float:
        if not self.examples:
            return 0.0
        return sum(e["self"] for e in self.examples) / len(self.examples)

    # ---------------- rendering ----------------
    def _render(self, input: str) -> tuple[str, str, bool]:
        if not self.examples:
            return "", "", False
        prior = self._prior()
        qtok = _tokens(input)
        scored = sorted(
            range(len(self.examples)),
            key=lambda i: -_sim(qtok, self.examples[i]["tok"]),
        )
        top = _sim(qtok, self.examples[scored[0]]["tok"])
        open_gate = top >= NEAR_DUP and (prior <= 1e-6 or top >= RATIO * prior)

        chosen: list[int] = []
        used = 0
        if open_gate:
            i = scored[0]
            blk = _clip(self.examples[i]["q"]) + "\n" + self.examples[i]["a"]
            chosen.append(i)
            used += len(blk) + 16
        precedent = ""
        if open_gate:
            e = self.examples[chosen[0]]
            precedent = (
                "NEAREST SOLVED CASE (very similar problem -- use its answer form):\n"
                f"Q: {_clip(e['q'])}\nA: {e['a']}\n\n"
            )

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

        header = "SOLVED CASES (one per answer category, rarest category first):\n\n"
        return header + "\n\n".join(blocks), precedent, open_gate

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        cases, precedent, gate = self._render(input)
        prompt = PROMPT_TEMPLATE.format(
            precedent=precedent, solved_cases=cases, input=input
        )
        response = self.call_llm(prompt)
        answer = extract_json_field(response, "final_answer")
        return answer, {
            "full_response": response,
            "gate_open": gate,
            "prior": round(self._prior(), 4),
            "num_examples": len(self.examples),
        }

    def get_context_length(self) -> int:
        if not self.examples:
            return 0
        c, p, _ = self._render(self.examples[len(self.examples) // 2]["q"])
        return len(c) + len(p)

    # ---------------- state ----------------
    def get_state(self) -> str:
        return json.dumps(
            {
                "examples": [
                    {"q": e["q"], "a": e["a"], "tok": sorted(e["tok"]), "self": e["self"]}
                    for e in self.examples
                ]
            },
            ensure_ascii=False,
        )

    def set_state(self, state: str) -> None:
        d = json.loads(state)
        self.examples = [
            {
                "q": e["q"],
                "a": e["a"],
                "tok": frozenset(e.get("tok", [])),
                "self": e.get("self", 0.0),
            }
            for e in d.get("examples", [])
        ]
