"""Answer-Signature Digest Memory.

Mechanism (new relative to demo-listing / pair-based memories): memory is not a
chronological list of solved items, it is an *answer inventory keyed by the
canonicalised answer string*.

  - Learning canonicalises each ground-truth answer (separator unification on
    ';', ',' and their full-width variants, component sorting, punctuation and
    'label:' style prefixes stripped, lower-cased) and merges all examples that
    map to the same signature.  So `a;b`, `b;a` and `A；B` collapse into one
    entry, and multi-component / composite answers become first-class memory
    entries with their own exemplars.
  - The digest keeps, per signature, a small spread of representative
    exemplars (first / middle / last of the bucket) so a signature that covers a
    wide range of situations is not represented by a single biased sample.
  - Entries are ordered by (frequency bucket, signature) so the injected digest
    is a *stable, compact inventory of the answer space* rather than a recency
    window.
  - Prediction shows the inventory and asks the solver to emit an answer whose
    canonical form appears in it, copying the separators exactly.

Rationale: with long-tail label sets the dominant error is producing a surface
form that no ground-truth answer ever has (wrong components, wrong separator,
wrong ordering). A signature-keyed inventory makes the *legal output space*
explicit in ~1/3 of the characters a raw demo bank needs, which should help
label-rich datasets while leaving easy small-label tasks untouched.
"""

import json
import re
from typing import Any

from ..llm import LLMCallable
from ..memory_system import MemorySystem, extract_json_field

PROMPT_TEMPLATE = """You must classify the problem below. The ANSWER INVENTORY lists every answer
value observed so far, with solved examples illustrating each one.

{inventory}

**Problem:**
{input}

**Instructions:**
- Pick the answer value from the inventory that best fits the problem
- Copy it EXACTLY as written (same components, same separator characters)
- If several components apply, join them with the separator used in the inventory and list them in a natural order
- Respond in JSON format

{{"reasoning": "[brief reasoning]", "final_answer": "[your answer]"}}"""

MAX_ANSWERS = 150
PER_ANSWER = 2
CTX_BUDGET = 26000
SNIPPET = 170

_SEP = re.compile(r"\s*[;；,，、]\s*")
_PREFIX = re.compile(r"^\s*(罪名|label|answer|final answer)\s*[:：]\s*", re.I)


def _canon_answer(a: str) -> str:
    s = str(a).strip()
    s = _PREFIX.sub("", s)
    s = _SEP.sub(";", s)
    parts = [p.strip().lower() for p in s.split(";") if p.strip()]
    if not parts:
        return " ".join(s.lower().split())
    # de-duplicate components while preserving a deterministic (sorted) order
    uniq = sorted(set(parts))
    return ";".join(uniq)


def _spread(members: list[dict[str, str]], k: int) -> list[dict[str, str]]:
    if len(members) <= k:
        return members
    if k <= 1:
        return [members[-1]]
    idx = [0]
    step = (len(members) - 1) / (k - 1)
    while len(idx) < k:
        nxt = min(len(members) - 1, int(round(idx[-1] + step)))
        if nxt <= idx[-1]:
            nxt = idx[-1] + 1
        idx.append(nxt)
    return [members[i] for i in idx[:k]]


class AnswerSignatureDigestMemory(MemorySystem):
    """Memory keyed by canonical answer signatures, rendered as an inventory."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self.groups: dict[str, list[dict[str, str]]] = {}

    # ---------------- learning ----------------
    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        for r in batch_results:
            ans = str(r["ground_truth"]).strip()
            if not ans:
                continue
            sig = _canon_answer(ans)
            q = " ".join(str(r.get("raw_question") or r["input"]).split())[:SNIPPET]
            self.groups.setdefault(sig, []).append({"question": q, "answer": ans})

    # ---------------- rendering ----------------
    def _render(self) -> str:
        if not self.groups:
            return ""
        items = sorted(self.groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))
        parts: list[str] = []
        total = 0
        for sig, members in items:
            if len(parts) >= MAX_ANSWERS:
                break
            lines = [f'ANSWER: "{sig}"  ({len(members)} solved example(s))']
            per = PER_ANSWER if len(members) > 1 else 1
            for m in _spread(members, per):
                lines.append(f"    e.g. {m['question']}")
            block = "\n".join(lines)
            if total + len(block) > CTX_BUDGET and parts:
                break
            parts.append(block)
            total += len(block) + 2
        if not parts:
            return ""
        return "ANSWER INVENTORY (signature -> illustrative solved examples):\n\n" + "\n\n".join(parts)

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        section = self._render()
        prompt = PROMPT_TEMPLATE.format(inventory=section, input=input)
        response = self.call_llm(prompt)
        answer = extract_json_field(response, "final_answer")
        return answer, {
            "full_response": response,
            "num_signatures": len(self.groups),
        }

    def get_context_length(self) -> int:
        return len(self._render())

    # ---------------- state ----------------
    def get_state(self) -> str:
        return json.dumps({"groups": self.groups}, indent=2, ensure_ascii=False)

    def set_state(self, state: str) -> None:
        data = json.loads(state)
        self.groups = data.get("groups", {}) or {}
