"""Contrastive Pair Memory.

Mechanism (new relative to demo-listing memories): the memory stores *pairs of
examples that share surface vocabulary but differ in their answer*, plus
*error pairs* when a previous prediction was wrong.

  - Learning keeps every solved example, but additionally mines a contrastive
    set: for each newly seen example, a lexical neighbour with a DIFFERENT
    answer is found and the two are stored as a contrastive pair. When a batch
    reports `was_correct == False` (available in online mode), the pair is
    flagged as an error pair and ranked first.
  - Prediction retrieves only the pairs whose "positive" member is
    lexically close to the query, and renders them as:

        A: <problem> -> <answer>
        B: <very similar problem> -> <different answer>
        Difference to look at: <the words only A has>

    i.e. the prompt is organised around *discriminating features* instead of a
    flat list of demonstrations. Up to TOP_DEMOS plain positives fill the rest.

Rationale: a bank of same-task demonstrations teaches the output format but not
the decision boundary; the boundary is what distinguishes confusable answers,
and it is exactly what a contrastive pair makes explicit while staying compact.
"""

import json
import re
from typing import Any

from ..llm import LLMCallable
from ..memory_system import MemorySystem, extract_json_field

PROMPT_TEMPLATE = """Solve the problem below. The contrastive pairs show two similar problems
with DIFFERENT answers; use the highlighted difference to decide which answer applies.

{pairs_section}

**Problem:**
{input}

**Instructions:**
- Follow the answer conventions of the examples
- Respond in JSON format

{{"reasoning": "[brief reasoning]", "final_answer": "[your answer]"}}"""

MAX_PAIRS = 6
MAX_POSITIVES = 3
CTX_BUDGET = 9000
SNIPPET = 340

_CJK = re.compile(r"[\u4e00-\u9fff]")
_STOP = set("the a an of and or is are to in on for with that this it as at by".split())


def _tokens(text: str) -> set[str]:
    low = text.lower()
    out: set[str] = set(w for w in re.findall(r"[a-z0-9]+", low) if w not in _STOP)
    extra: set[str] = set()
    for n in (4, 5):
        for i in range(len(low) - n + 1):
            extra.add(low[i : i + n])
    cjk = _CJK.findall(low)
    for n in (2, 3):
        for i in range(len(cjk) - n + 1):
            extra.add("".join(cjk[i : i + n]))
    return out | extra


def _similarity(a: set[str], b: set[str]) -> float:
    inter = a & b
    if not inter:
        return 0.0
    return len(inter) / ((len(a) * len(b)) ** 0.5 + 1e-9)


def _norm_answer(a: str) -> str:
    return " ".join(str(a).strip().lower().split())


class ContrastivePairMemory(MemorySystem):
    """Memory organised as same-vocabulary / different-answer pairs."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self.examples: list[dict[str, Any]] = []  # {snippet, answer, toks}
        self.pairs: list[dict[str, Any]] = []  # {pos_idx, neg_idx, is_error}

    # ---------------- learning ----------------
    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        for r in batch_results:
            raw = " ".join(str(r.get("raw_question") or r["input"]).split())[:SNIPPET]
            answer = str(r["ground_truth"]).strip()
            toks = _tokens(raw)
            idx = len(self.examples)
            self.examples.append({"snippet": raw, "answer": answer})
            was_correct = r.get("was_correct")
            is_error = (was_correct is False) or (
                was_correct is True
                and _norm_answer(r.get("prediction", "")) != _norm_answer(answer)
            )

            best_i, best_s = None, 0.0
            for j in range(idx - 1, max(-1, idx - 200), -1):
                other = self.examples[j]
                if _norm_answer(other["answer"]) == _norm_answer(answer):
                    continue
                s = _similarity(toks, _tokens(other["snippet"]))
                if s > best_s:
                    best_i, best_s = j, s
            if best_i is not None and best_s > 0.05:
                self.pairs.append(
                    {"pos": idx, "neg": best_i, "is_error": bool(is_error), "sim": best_s}
                )
                if len(self.pairs) > 600:
                    # keep error pairs first, then the most similar ones
                    self.pairs.sort(key=lambda p: (p["is_error"], p["sim"]), reverse=True)
                    self.pairs = self.pairs[:500]

    # ---------------- retrieval ----------------
    def _render(self, input: str) -> str:
        if not self.examples:
            return ""
        qtok = _tokens(input)
        scored = []
        for p in self.pairs:
            pos = self.examples[p["pos"]]
            s = _similarity(qtok, _tokens(pos["snippet"]))
            scored.append((p["is_error"], s, p))
        scored.sort(key=lambda x: (not x[0], -x[1]))

        used_pos: set[int] = set()
        parts, total = [], 0
        for _, s, p in scored:
            if len(parts) >= MAX_PAIRS:
                break
            if p["pos"] in used_pos:
                continue
            pos, neg = self.examples[p["pos"]], self.examples[p["neg"]]
            a_tok, b_tok = _tokens(pos["snippet"]), _tokens(neg["snippet"])
            diff = [t for t in list(a_tok - b_tok) if len(t) <= 24][:6]
            block = (
                f"PAIR\nA: {pos['snippet']}\nA-answer: {pos['answer']}\n"
                f"B (similar problem, different answer): {neg['snippet']}\n"
                f"B-answer: {neg['answer']}\n"
                f"Difference to look at: A mentions {', '.join(diff) if diff else 'other details'}"
            )
            if total + len(block) > CTX_BUDGET and parts:
                break
            parts.append(block)
            used_pos.add(p["pos"])
            total += len(block) + 2

        for rec in reversed(self.examples):
            if len(parts) >= MAX_PAIRS + MAX_POSITIVES:
                break
            if rec["snippet"] in {self.examples[i]["snippet"] for i in used_pos}:
                continue
            block = f"EXAMPLE\nQ: {rec['snippet']}\nA: {rec['answer']}"
            if total + len(block) > CTX_BUDGET:
                break
            parts.append(block)
            total += len(block) + 2

        joined = "\n\n".join(parts)
        if not joined:
            return ""
        return "CONTRASTIVE PAIRS:\n\n" + joined

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        section = self._render(input)
        prompt = PROMPT_TEMPLATE.format(pairs_section=section, input=input)
        response = self.call_llm(prompt)
        answer = extract_json_field(response, "final_answer")
        return answer, {
            "full_response": response,
            "num_pairs": len(self.pairs),
            "num_examples": len(self.examples),
        }

    def get_context_length(self) -> int:
        if not self.examples:
            return 0
        return len(self._render(self.examples[len(self.examples) // 2]["snippet"]))

    # ---------------- state ----------------
    def get_state(self) -> str:
        return json.dumps(
            {"examples": self.examples, "pairs": self.pairs}, indent=2
        )

    def set_state(self, state: str) -> None:
        data = json.loads(state)
        self.examples = data.get("examples", [])
        self.pairs = data.get("pairs", [])
