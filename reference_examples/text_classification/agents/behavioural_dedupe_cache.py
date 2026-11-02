"""Behavioural Dedupe Cache (exact-chain memory).

Mechanism (new relative to demo-listing memories): the memory treats past
(prediction, ground_truth) pairs as *behavioural duplicates* instead of as
demo material.

  - Learning records every solved example twice: as a compact (input-key,
    gold-answer) association, and as a *duplicate group key* derived from the
    canonicalised answer (atom maps / explicit-H brackets / parenthesisation
    normalised away, fragments sorted). All gold answers sharing a canonical
    key are merged into ONE representative answer string.
  - Prediction: a lexical neighbour search finds near-identical past inputs; if
    a neighbour is an exact duplicate of the query, its stored answer is returned
    directly without an LLM call. Otherwise the prompt receives a short
    "duplicate resolution" note listing canonical keys that previously mapped to
    several surface forms, together with the canonical form to emit.

Rationale: when the answer space is open (strings, identifiers, multi-label
combinations) two thirds of the memory is spent on inputs whose answer is
already known; searching the behaviour (canonicalised answers) both trims the
context and gives the solver an output-convention hint that raw demos never
state explicitly.
"""

import json
import re
from typing import Any

from ..llm import LLMCallable
from ..memory_system import MemorySystem, extract_json_field

PROMPT_TEMPLATE = """Solve the problem below using the resolved duplicate records as format guidance.

{dup_section}

**Problem:**
{input}

**Instructions:**
- Emit the answer in exactly the surface form shown by the records above
- Respond in JSON format

{{"reasoning": "[brief reasoning]", "final_answer": "[your answer]"}}"""

MAX_RECORDS = 12
CTX_BUDGET = 5000
EXACT_SIM = 0.92

_MAP_RE = re.compile(r":\d+|%\d\d")
_BRACKET_RE = re.compile(r"\[[^]]*H[^]]*\]")
_CJK = re.compile(r"[\u4e00-\u9fff]")


def _fragments(answer: str) -> list[str]:
    parts = re.split(r"[.;；,，、]", str(answer))
    return [p.strip() for p in parts if p.strip()]


def _canon_token(tok: str) -> str:
    t = _MAP_RE.sub("", tok)
    t = _BRACKET_RE.sub("", t)
    t = t.replace("@@", "@")
    t = t.replace("(", "").replace(")", "")
    return t.strip().lower()


def canonical_key(answer: str) -> str:
    """Answer-space key: order-free, decorator-free rendering of an answer."""
    toks = sorted(_canon_token(f) for f in _fragments(answer))
    return "|".join(t for t in toks if t)


def _tokens(text: str) -> set[str]:
    low = text.lower()
    out: set[str] = set(re.findall(r"[a-z0-9]+", low))
    uniq = set(out)
    for n in (4, 5):
        for i in range(len(low) - n + 1):
            uniq.add(low[i : i + n])
    cjk = _CJK.findall(low)
    for n in (2, 3):
        for i in range(len(cjk) - n + 1):
            uniq.add("".join(cjk[i : i + n]))
    return uniq


class BehaviouralDedupeCache(MemorySystem):
    """Memory keyed on canonicalised answers; exact duplicates bypass the LLM."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self.records: list[dict[str, str]] = []  # {qkey, answer, canon}
        self.groups: dict[str, str] = {}  # canon -> representative answer

    # ---------------- learning ----------------
    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        for r in batch_results:
            raw = str(r.get("raw_question") or r["input"])
            answer = str(r["ground_truth"]).strip()
            canon = canonical_key(answer)
            if not canon:
                continue
            self.groups.setdefault(canon, answer)
            qkey = " ".join(raw.split())[:400]
            if any(rec["qkey"] == qkey for rec in self.records):
                continue
            self.records.append({"qkey": qkey, "answer": answer, "canon": canon})

    # ---------------- retrieval ----------------
    def _neighbours(self, input: str) -> list[tuple[float, dict[str, str]]]:
        if not self.records:
            return []
        qtok = _tokens(input)
        if not qtok:
            return []
        scored = []
        for rec in self.records[-4000:]:
            toks = _tokens(rec["qkey"])
            inter = qtok & toks
            if not inter:
                continue
            s = len(inter) / ((len(qtok) * len(toks)) ** 0.5 + 1e-9)
            scored.append((s, rec))
        scored.sort(key=lambda x: -x[0])
        return scored[:MAX_RECORDS]

    def _dup_section(self, input: str) -> str:
        nb = self._neighbours(input)
        if not nb:
            return ""
        parts, total = [], 0
        multi = 0
        for s, rec in nb:
            others = [r["answer"] for r in self.records if r["canon"] == rec["canon"]]
            distinct = sorted(set(others))
            if len(distinct) > 1:
                multi += 1
                line = (
                    f"- canonical form: [{distinct[0]}] "
                    f"(seen as {len(distinct)} surface variants) "
                    f"| closest problem: {rec['qkey'][:160]}"
                )
            else:
                line = (
                    f"- resolved answer: [{rec['answer'][:160]}] "
                    f"| closest problem: {rec['qkey'][:160]}"
                )
            if total + len(line) > CTX_BUDGET and parts:
                break
            parts.append(line)
            total += len(line) + 1
        if not parts:
            return ""
        header = (
            "RESOLVED DUPLICATE RECORDS (answers already known for near-identical "
            "problems; reuse the exact surface form):\n"
        )
        return header + "\n".join(parts)

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        nb = self._neighbours(input)
        if nb and nb[0][0] >= EXACT_SIM:
            return nb[0][1]["answer"], {
                "full_response": "",
                "cache_hit": True,
                "sim": nb[0][0],
            }
        section = self._dup_section(input)
        prompt = PROMPT_TEMPLATE.format(dup_section=section, input=input)
        response = self.call_llm(prompt)
        answer = extract_json_field(response, "final_answer")
        return answer, {
            "full_response": response,
            "cache_hit": False,
            "num_records": len(self.records),
        }

    def get_context_length(self) -> int:
        if not self.records:
            return 0
        return len(self._dup_section(self.records[len(self.records) // 2]["qkey"]))

    # ---------------- state ----------------
    def get_state(self) -> str:
        return json.dumps({"records": self.records, "groups": self.groups}, indent=2)

    def set_state(self, state: str) -> None:
        data = json.loads(state)
        self.records = data.get("records", [])
        self.groups = data.get("groups", {})
