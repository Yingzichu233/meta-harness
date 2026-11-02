"""Straddle Coverage Bank.

Two structural changes relative to the strongest baseline (which injects the
whole arrival-ordered bank clipped to a char cap):

  1. *Selection* is set-cover greedy instead of arrival order.  The budget is
     filled by repeatedly taking the demonstration whose text tokens are least
     covered by what is already in the prompt, and every distinct answer gets a
     guaranteed slot first (rarest first).  The result is a bank that spends its
     characters on *information* rather than on near-duplicates of the records
     that happened to arrive early.

  2. *Clipping* is straddle-based.  A demo is shown as its head + tail, where
     the tail length is derived from the median record length in the memory:
     when records are long, structure tends to sit at the end of the record, so
     a head-only clip silently deletes it.  Short records are kept verbatim.

Additionally, the prompt carries a compact `ANSWER VOCABULARY` line listing every
distinct answer seen so far (cheap, ~1 short line) so long-tail answer spaces are
visible even when their demo did not make the budget.

Rationale: on long-tail answer spaces the baseline never shows most classes; on
tasks with long records it truncates away the informative suffix.  Fixing both
is purely structural and keeps the exact prompt layout that wins today.
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


class StraddleCoverageBank(MemorySystem):
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
        covered: set[str] = set()

        # pass 1: one representative per distinct answer, rarest first
        counts: dict[str, int] = {}
        order: list[str] = []
        for e in self.examples:
            k = e["a"].lower()
            if k not in counts:
                counts[k] = 0
                order.append(k)
            counts[k] += 1
        first_of: dict[str, int] = {}
        for i, e in enumerate(self.examples):
            first_of.setdefault(e["a"].lower(), i)
        for k in sorted(order, key=lambda k: counts[k]):
            i = first_of[k]
            if used + cost[i] > CTX_BUDGET or len(chosen) >= MAX_DEMOS:
                continue
            chosen.append(i)
            used += cost[i]
            covered |= toks[i]

        # pass 2: greedy marginal-coverage fill
        while len(chosen) < MAX_DEMOS:
            best_i, best_gain = -1, 0
            for i in range(n):
                if i in chosen or used + cost[i] > CTX_BUDGET:
                    continue
                gain = len(toks[i] - covered)
                if gain > best_gain:
                    best_gain, best_i = gain, i
            if best_i < 0:
                break
            chosen.append(best_i)
            used += cost[best_i]
            covered |= toks[best_i]
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
