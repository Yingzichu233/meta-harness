"""Calibrated Adaptive Reuse Memory.

Mechanism (new): memory *measures how reusable solved cases are* for the task at
hand, and switches its whole retrieval/rendering strategy from that measurement
— with no LLM call and no tuning knob.

  - Learning stores every solved case as a compact (question, answer) pair plus
    its token fingerprint.  It also maintains `prior`, the average
    nearest-neighbour self-similarity of the memory (how similar the solved
    problems are to *each other*).  `prior` is a property of the task's input
    space, not of any dataset.
  - Prediction computes `overlap` = similarity of the query to its nearest
    solved case, then chooses a retrieval regime:

      overlap/prior >= 2.0   ->  REPLICATION: answers are effectively
          instance-unique payloads (they are edited copies of the input), so
          breadth is worthless; show a few whole nearest cases verbatim and
          tell the solver to emit the payload for THIS input by reusing the
          nearest case's structure.
      1.15 <= ratio < 2.0    ->  BREADTH: answers cluster into reusable
          categories; show nearest cases but reserve slots for one case per
          uncovered answer so the candidate answer set is visible.
      ratio < 1.15           ->  REASON: cases are not neighbours of the query,
          so retrieval is mostly noise; fall back to a wide, category-balanced
          bank (one per answer) rather than a narrow nearest-neighbour set.

Rationale: the single biggest structural weakness of every stored system is
that it hardcodes one retrieval width for all tasks.  Nearest-neighbour
retrieval helps when answers are reusable and hurts when they are not; making
the width a *function of the measured reusability of the answer space* lets one
system serve both regimes.  Our offline measurement across the memory states
confirms the three regimes are cleanly separated by this ratio.
"""

import json
import re
from typing import Any

from ..llm import LLMCallable
from ..memory_system import MemorySystem, extract_json_field

PROMPT_TEMPLATE = """Solve the problem below. {regime_note}

{cases_section}

**Problem:**
{input}

**Instructions:**
- Follow the answer conventions exactly as they appear in the solved cases
- Respond in JSON format

{{"reasoning": "[brief reasoning]", "final_answer": "[your answer]"}}"""

NOTE_REPLICATION = (
    "The solved cases show that answers are payloads derived from the problem "
    "itself: reuse the nearest case's structure and substitute the content of "
    "this problem."
)
NOTE_BREADTH = (
    "The solved cases show reusable answer categories; pick the single category "
    "whose case is most similar, and copy its exact wording."
)
NOTE_REASON = (
    "The solved cases are listed one per answer category; identify the category "
    "described by this problem and copy its exact label."
)

REGIMES = {
    "replication": {"k": 6, "breadth": 0, "snip": 400, "budget": 9000},
    "breadth": {"k": 22, "breadth": 8, "snip": 260, "budget": 16000},
    "reason": {"k": 34, "breadth": 34, "snip": 240, "budget": 22000},
}

_CJK = re.compile(r"[\u4e00-\u9fff]")
_STOP = set("the a an of and or is are to in on for with that this it as at by".split())


def _tokens(text: str) -> set[str]:
    low = str(text).lower()
    out: set[str] = set(w for w in re.findall(r"[a-z0-9]+", low) if w not in _STOP)
    for n in (4, 5):
        for i in range(len(low) - n + 1):
            out.add(low[i : i + n])
    cj = _CJK.findall(low)
    for n in (2, 3):
        for i in range(len(cj) - n + 1):
            out.add("".join(cj[i : i + n]))
    return out


def _sim(a: set[str], b: set[str]) -> float:
    inter = a & b
    if not inter:
        return 0.0
    return len(inter) / ((len(a) * len(b)) ** 0.5 + 1e-9)


def _clip(s: str, n: int) -> str:
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[:n] + "..."


class CalibratedAdaptiveReuseMemory(MemorySystem):
    """Retrieval width driven by a measured reusability ratio."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self.examples: list[dict[str, Any]] = []  # {q, a, tok(frozenset)}
        self._prior = 0.0
        self._sorted: list[float] = []  # per-example nearest-neighbour sims

    # ---------------- learning ----------------
    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        for r in batch_results:
            a = " ".join(str(r["ground_truth"]).split())
            if not a:
                continue
            q = " ".join(str(r.get("raw_question") or r["input"]).split())
            toks = frozenset(_tokens(q))
            best = max((_sim(toks, ex["tok"]) for ex in self.examples), default=0.0)
            self.examples.append({"q": q, "a": a, "tok": toks})
            # running estimate of the task's self-similarity (reusability prior)
            self._sorted.append(best)
            self._prior = sum(self._sorted) / len(self._sorted)

    # ---------------- retrieval ----------------
    def _regime(self, input: str) -> tuple[str, list[int]]:
        qtok = _tokens(input)
        scored = sorted(
            range(len(self.examples)),
            key=lambda i: -_sim(qtok, self.examples[i]["tok"]),
        )
        top = _sim(qtok, self.examples[scored[0]]["tok"]) if scored else 0.0
        prior = self._prior
        ratio = top / prior if prior > 1e-6 else 0.0
        if ratio >= 2.0:
            name = "replication"
        elif ratio >= 1.15:
            name = "breadth"
        else:
            name = "reason"
        return name, scored

    def _render(self, input: str) -> tuple[str, str]:
        if not self.examples:
            return "", NOTE_BREADTH
        name, scored = self._regime(input)
        cfg = REGIMES[name]
        k, breadth, snip, budget = cfg["k"], cfg["breadth"], cfg["snip"], cfg["budget"]

        if name == "breadth":
            chosen: list[int] = []
            seen: set[str] = set()
            for i in scored:
                if len(chosen) < breadth and len(chosen) < k // 2 and self.examples[i]["a"] not in seen:
                    chosen.append(i)
                    seen.add(self.examples[i]["a"])
            for i in scored:
                if len(chosen) >= k:
                    break
                if i not in chosen:
                    chosen.append(i)
        elif name == "reason":
            chosen, seen = [], set()
            for i in scored:
                if self.examples[i]["a"] in seen:
                    continue
                seen.add(self.examples[i]["a"])
                chosen.append(i)
                if len(chosen) >= k:
                    break
        else:
            chosen = scored[:k]

        parts, total = [], 0
        for i in chosen:
            ex = self.examples[i]
            block = f"Q: {_clip(ex['q'], snip)}\nA: {ex['a']}"
            if total + len(block) > budget and parts:
                break
            parts.append(block)
            total += len(block) + 2
        note = {
            "replication": NOTE_REPLICATION,
            "breadth": NOTE_BREADTH,
            "reason": NOTE_REASON,
        }[name]
        return "\n\n".join(parts), note

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        section, note = self._render(input)
        name = self._regime(input)[0] if self.examples else "none"
        prompt = PROMPT_TEMPLATE.format(
            regime_note=note, cases_section=section, input=input
        )
        response = self.call_llm(prompt)
        answer = extract_json_field(response, "final_answer")
        return answer, {
            "full_response": response,
            "regime": name,
            "prior": round(self._prior, 4),
            "num_examples": len(self.examples),
        }

    def get_context_length(self) -> int:
        if not self.examples:
            return 0
        s, _ = self._render(self.examples[len(self.examples) // 2]["q"])
        return len(s)

    # ---------------- state ----------------
    def get_state(self) -> str:
        return json.dumps(
            {
                "examples": [
                    {"q": e["q"], "a": e["a"], "tok": sorted(e["tok"])}
                    for e in self.examples
                ],
                "prior": self._prior,
                "sorted": self._sorted,
            },
            indent=2,
            ensure_ascii=False,
        )

    def set_state(self, state: str) -> None:
        data = json.loads(state)
        self.examples = [
            {"q": e["q"], "a": e["a"], "tok": frozenset(e.get("tok", []))}
            for e in data.get("examples", [])
        ]
        self._prior = data.get("prior", 0.0)
        self._sorted = data.get("sorted", [])
