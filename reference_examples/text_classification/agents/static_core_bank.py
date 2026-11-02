"""Static Core Bank — decoupling *what is shown* from *when it is shown*.

Diagnosis (measured over the stored memories of all three task regimes): the
information a demonstration bank carries is overwhelmingly *time-invariant*.
Using the FULL memory as a stand-in for the best possible bank, plain
nearest-neighbour retrieval already reaches ~74% top-1 / 88% top-5 on a
22-class task, and one-example-per-label set cover reaches ~96% -- both
computable from the first few dozen cases.  Yet every stored system
re-renders and re-retrieves a *growing* bank on every query, so its context
composition drifts as cases accumulate and its per-query cost is O(|memory|).

Mechanism:
  * Learning only appends cases and marks the structure dirty on a *doubling
    schedule* (after 1, 3, 7, 15, ... cases), so the bank is rebuilt O(log n)
    times rather than once per query.
  * The bank is a *static core*: one representative per distinct answer
    (rarest answer first) so a long-tail answer space is fully visible, then,
    **only if the task's queries are measured to be near-duplicates of each
    other** (mean nearest-neighbour self-similarity of the memory above a
    fixed absolute level), anchored nearest-neighbour fill — more exemplars of
    the answers that keep recurring.  Self-similarity is a property of the
    task's input space, maintained online; no dataset-specific knob.
  * Prediction prepends the core verbatim and makes no retrieval decision: the
    identical context is used for every query.

Rationale: growth-driven context drift is a plausible source of the unstable
val behaviour we see, and a static bank costs nothing when the cases it omits
are duplicates of ones it keeps.
"""

import json
import re
from typing import Any

from ..llm import LLMCallable
from ..memory_system import MemorySystem, extract_json_field

PROMPT_TEMPLATE = """Solve the problem below based on the solved cases provided.

{solved_cases}

**Problem:**
{input}

**Instructions:**
- Identify which solved case (or which answer category) this problem belongs to
- Answer exactly in the form the solved cases use
- Respond in JSON format

{{"reasoning": "[brief reasoning]", "final_answer": "[your answer]"}}"""

CTX_BUDGET = 26000      # characters of solved cases
SNIP = 200              # chars per case (records longer than this are tail-clipped)
NEAR_DUP = 0.60         # absolute self-similarity level above which we add
                        # extra exemplars of recurring answers
ANCHOR_MARGIN = 1.25    # extra exemplar must be this*prior similar to a kept case
MAX_CASES = 120

_CJK = re.compile(r"[\u4e00-\u9fff]")
_STOP = set("the a an of and or is are to in on for with that this it as at by".split())


def _tokens(text: str) -> set[str]:
    low = " ".join(str(text).lower().split())
    out = {w for w in re.findall(r"[a-z0-9]+", low) if w not in _STOP}
    for n in (4, 5):                      # ascii character n-grams too: needed
        for i in range(len(low) - n + 1):  # for dense/chemical tokens
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


def _ws(t: str) -> str:
    return " ".join(str(t).split())


class StaticCoreBank(MemorySystem):
    """A frozen, answer-covering demo bank rebuilt only on a doubling schedule."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self.examples: list[dict[str, Any]] = []   # {q, a, tok}
        self.core: list[int] = []
        self._dirty_at = 1                          # rebuild threshold
        self._bank = ""                             # cached rendered bank
        self._prior = 0.0                           # task self-similarity

    # ---------------- learning ----------------
    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        for r in batch_results:
            a = _ws(r["ground_truth"])
            if not a:
                continue
            q = _ws(r.get("raw_question") or r["input"])
            tok = frozenset(_tokens(q))
            best = max((_sim(tok, ex["tok"]) for ex in self.examples), default=0.0)
            self.examples.append({"q": q, "a": a, "tok": tok, "self": best})
        n = len(self.examples)
        if n >= self._dirty_at:                     # doubling schedule
            self._dirty_at = max(self._dirty_at * 2 + 1, 1)
            self._rebuild()

    # ---------------- bank construction ----------------
    def _rebuild(self) -> None:
        ex = self.examples
        n = len(ex)
        if not n:
            self.core, self._bank = [], ""
            return
        self._prior = sum(e["self"] for e in ex) / n

        def clip(s: str) -> str:
            s = _ws(s)
            return s if len(s) <= SNIP else s[: SNIP // 2] + " ... " + s[-SNIP // 2 :]

        cost = [len(clip(e["q"])) + len(e["a"]) + 8 for e in ex]

        # pass 1: one representative per distinct answer, rarest answer first
        counts: dict[str, int] = {}
        for e in ex:
            counts[e["a"]] = counts.get(e["a"], 0) + 1
        first: dict[str, int] = {}
        for i, e in enumerate(ex):
            first.setdefault(e["a"], i)
        chosen: list[int] = []
        used = 0
        for ans in sorted(first, key=lambda k: counts[k]):
            i = first[ans]
            if used + cost[i] > CTX_BUDGET or len(chosen) >= MAX_CASES:
                continue
            chosen.append(i)
            used += cost[i]

        # pass 2: ONLY if queries are near-duplicates of each other, add more
        # exemplars anchored to already-kept cases.
        if self._prior >= NEAR_DUP:
            thresh = ANCHOR_MARGIN * self._prior
            while len(chosen) < MAX_CASES and used < CTX_BUDGET:
                best_i, best_g = -1, 0.0
                kept = set(chosen)
                for i in range(n):
                    if i in kept or used + cost[i] > CTX_BUDGET:
                        continue
                    g = max((_sim(ex[i]["tok"], ex[j]["tok"]) for j in kept), default=0.0)
                    if g > best_g and g > thresh:
                        best_g, best_i = g, i
                if best_i < 0:
                    break
                chosen.append(best_i)
                used += cost[best_i]

        self.core = sorted(chosen)
        blocks = [f"Q: {clip(ex[i]['q'])}\nA: {ex[i]['a']}" for i in self.core]
        self._bank = (
            "SOLVED CASES (one per answer category%s):\n\n" % (
                ", plus further examples of recurring categories" if self._prior >= NEAR_DUP else ""
            )
            + "\n\n".join(blocks)
        )

    # ---------------- prediction ----------------
    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        if not self._bank:
            self._rebuild()
        prompt = PROMPT_TEMPLATE.format(solved_cases=self._bank, input=input)
        response = self.call_llm(prompt)
        answer = extract_json_field(response, "final_answer")
        return answer, {
            "full_response": response,
            "num_cases_shown": len(self.core),
            "prior": round(self._prior, 4),
            "num_examples": len(self.examples),
        }

    def get_context_length(self) -> int:
        if not self._bank:
            self._rebuild()
        return len(self._bank)

    # ---------------- state ----------------
    def get_state(self) -> str:
        return json.dumps(
            {
                "examples": [
                    {"q": e["q"], "a": e["a"], "tok": sorted(e["tok"]), "self": e["self"]}
                    for e in self.examples
                ],
                "core": self.core,
                "dirty_at": self._dirty_at,
                "prior": self._prior,
            },
            ensure_ascii=False,
        )

    def set_state(self, state: str) -> None:
        d = json.loads(state)
        self.examples = [
            {"q": e["q"], "a": e["a"], "tok": frozenset(e.get("tok", [])), "self": e.get("self", 0.0)}
            for e in d.get("examples", [])
        ]
        self.core = d.get("core", [])
        self._dirty_at = d.get("dirty_at", max(len(self.examples) * 2 + 1, 1))
        self._prior = d.get("prior", 0.0)
        self._rebuild()
