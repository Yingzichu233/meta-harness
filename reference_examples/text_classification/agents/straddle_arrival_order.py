"""ABLATION of StraddleCoverageBank: arrival-order selection.

Identical to agents/straddle_coverage_bank.py EXCEPT that `_plan()` fills the
context budget in plain arrival order (like the few-shot baseline) instead of
the rarest-first + greedy marginal-coverage selection. Straddle clipping and
the ANSWER VOCABULARY line are kept unchanged, so any performance difference
isolates the contribution of the selection mechanism.
"""

import json
import re
from typing import Any

from ..llm import LLMCallable
from ..memory_system import MemorySystem, extract_json_field

PROMPT_TEMPLATE = """Solve the problem below based on the examples provided.

{examples_section}
{vocab_section}
**Problem:**
{input}

**Instructions:**
- First decide which kind of problem this is, using the examples and the answer vocabulary
- Then produce the answer following the conventions shown in the examples
- Respond in JSON format

{{"reasoning": "[brief reasoning]", "final_answer": "[your answer]"}}"""

CTX_BUDGET = 26000      # characters of demonstrations
SNIP = 220              # max total characters per demo snippet (head + tail)
HEAD_SHARE = 0.75       # share of the snippet budget given to the head
MAX_DEMOS = 96
VOCAB_BUDGET = 1800
MAX_VOCAB = 120


def _whitespace(t: str) -> str:
    return " ".join(str(t).split())


def _tokens(text: str) -> set[str]:
    low = str(text).lower()
    out = set(re.findall(r"[a-z0-9]{2,}", low))
    cj = re.findall(r"[\u4e00-\u9fff]", low)
    for n in (2, 3):
        for i in range(len(cj) - n + 1):
            out.add("".join(cj[i : i + n]))
    return out


def _straddle(text: str, head: int, tail: int) -> str:
    s = _whitespace(text)
    if len(s) <= head + tail + 5:
        return s
    return s[:head] + " ... " + s[-tail:]


class StraddleArrivalOrder(MemorySystem):
    """Set-cover ordered demonstration bank with head+tail clipping."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self.examples: list[dict[str, str]] = []  # {q, a}

    # ---------------- learning ----------------
    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        for r in batch_results:
            a = _whitespace(r["ground_truth"])
            if not a:
                continue
            q = _whitespace(r.get("raw_question") or r["input"])
            self.examples.append({"q": q, "a": a})

    # ---------------- selection ----------------
    def _clip_params(self) -> tuple[int, int]:
        lens = sorted(len(e["q"]) for e in self.examples)
        med = lens[len(lens) // 2] if lens else SNIP
        if med <= SNIP:
            return SNIP, 0
        head = int(SNIP * HEAD_SHARE)
        tail = max(16, SNIP - head)
        return head, tail

    def _plan(self) -> list[int]:
        n = len(self.examples)
        if not n:
            return []
        head, tail = self._clip_params()
        texts = [_straddle(e["q"], head, tail) for e in self.examples]
        toks = [_tokens(t) for t in texts]
        cost = [len(texts[i]) + len(self.examples[i]["a"]) + 8 for i in range(n)]

        chosen: list[int] = []
        used = 0
        # ABLATION: plain arrival order instead of rarest-first + set-cover fill.
        for i in range(n):
            if used + cost[i] > CTX_BUDGET or len(chosen) >= MAX_DEMOS:
                continue
            chosen.append(i)
            used += cost[i]
        return chosen

    def _render(self) -> tuple[str, str]:
        idx = self._plan()
        if not idx:
            return "", ""
        head, tail = self._clip_params()
        blocks = []
        for i in idx:
            e = self.examples[i]
            blocks.append(f"Q: {_straddle(e['q'], head, tail)}\nA: {e['a']}")
        # answer vocabulary: distinct answers seen so far, most frequent first
        counts: dict[str, int] = {}
        for e in self.examples:
            counts[e["a"]] = counts.get(e["a"], 0) + 1
        vocab_items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        vocab, total = [], 0
        for a, _c in vocab_items[:MAX_VOCAB]:
            if total + len(a) + 2 > VOCAB_BUDGET and vocab:
                break
            vocab.append(a)
            total += len(a) + 2
        vocab_section = (
            "\nANSWER VOCABULARY (every answer seen so far; prefer these exact forms):\n"
            + " | ".join(vocab)
            + "\n"
            if vocab
            else ""
        )
        return "\n\n".join(blocks), vocab_section

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        section, vocab = self._render()
        prompt = PROMPT_TEMPLATE.format(
            examples_section=section, vocab_section=vocab, input=input
        )
        response = self.call_llm(prompt)
        answer = extract_json_field(response, "final_answer")
        return answer, {
            "full_response": response,
            "num_examples": len(self.examples),
        }

    def get_context_length(self) -> int:
        s, v = self._render()
        return len(s) + len(v)

    # ---------------- state ----------------
    def get_state(self) -> str:
        return json.dumps({"examples": self.examples}, indent=2, ensure_ascii=False)

    def set_state(self, state: str) -> None:
        self.examples = json.loads(state).get("examples", [])
