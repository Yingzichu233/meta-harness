"""Class-Balanced Exemplar Bank.

Mechanism (new relative to existing memories): the memory is a *coverage-ordered
exemplar bank*.  Instead of appending demos in arrival order and truncating by a
character cap (which lets the first-seen raw examples — usually the longest and
most redundant ones — eat the whole budget), learning re-orders memory by a
class-coverage priority:

  1. every solved example is bucketed by its exact answer string;
  2. one representative per answer is kept *first* (rarest answers first, so a
     long tail of rare classes is never crowded out by frequent ones);
  3. remaining budget is filled by a second pass that adds a second (then
     third) example to the answers that have spares — balancing classes;
  4. per-example snippets are whitespace-collapsed and clipped, so one long
     verbose record cannot monopolise the bank.

Prediction injects that balanced prefix verbatim in a plain Q/A layout (the
layout the strongest baseline uses), so the only thing that changed relative to
the baseline is *which* demonstrations occupy the budget: maximum label coverage
instead of maximum recency/length.

Rationale: raw banks waste context on duplicates of frequent answers, so the
solver sees almost no rare classes; coverage-first ordering is a purely
structural change that should help label-rich long-tail tasks without hurting
small-label tasks.
"""

import json
import re
from typing import Any

from ..llm import LLMCallable
from ..memory_system import MemorySystem, extract_json_field

PROMPT_TEMPLATE = """Solve the problem below based on the examples provided.

{examples_section}

**Problem:**
{input}

**Instructions:**
- Follow the patterns shown in the examples above
- Respond in JSON format

{{"reasoning": "[your reasoning]", "final_answer": "[your answer]"}}"""

CORE_DEMOS = 32        # guaranteed coverage slots (one per distinct answer)
MAX_DEMOS = 72         # hard cap on injected demonstrations
CTX_BUDGET = 26000     # characters of demonstrations injected per query
SNIPPET = 320          # per-demo snippet clip


def _norm_answer(a: str) -> str:
    return " ".join(str(a).strip().lower().split())


def _snippet(text: str, limit: int = SNIPPET) -> str:
    s = " ".join(str(text).split())
    return s[:limit]


class BalancedExemplarBank(MemorySystem):
    """Coverage-ordered, class-balanced demonstration bank."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self.examples: list[dict[str, str]] = []  # {question, answer}

    # ---------------- learning ----------------
    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        for r in batch_results:
            ans = str(r["ground_truth"]).strip()
            if not ans:
                continue
            q = _snippet(r.get("raw_question") or r["input"])
            self.examples.append({"question": q, "answer": ans})

    # ---------------- selection ----------------
    def _plan(self) -> list[dict[str, str]]:
        """Return the coverage-ordered list of demos to inject (cached order)."""
        if not self.examples:
            return []
        buckets: dict[str, list[dict[str, str]]] = {}
        order: list[str] = []
        for ex in self.examples:
            key = _norm_answer(ex["answer"])
            if key not in buckets:
                buckets[key] = []
                order.append(key)
            buckets[key].append(ex)

        # primary pass: one representative per distinct answer, rarest first
        primary = sorted(order, key=lambda k: (len(buckets[k]), k))
        plan: list[dict[str, str]] = []
        pointers = {k: 0 for k in primary}
        for k in primary:
            plan.append(buckets[k][0])
            pointers[k] = 1
            if len(plan) >= CORE_DEMOS:
                break

        # secondary pass: round-robin over answers that still have spares
        remaining = [
            k for k in primary
            if pointers[k] < len(buckets[k])
        ]
        while len(plan) < MAX_DEMOS:
            added = False
            for k in remaining:
                if pointers[k] < len(buckets[k]):
                    plan.append(buckets[k][pointers[k]])
                    pointers[k] += 1
                    added = True
                    if len(plan) >= MAX_DEMOS:
                        break
            if not added:
                break
        return plan

    def _render(self) -> str:
        plan = self._plan()
        parts: list[str] = []
        total = 0
        for ex in plan:
            block = f"Q: {ex['question']}\nA: {ex['answer']}"
            if total + len(block) > CTX_BUDGET and parts:
                break
            parts.append(block)
            total += len(block) + 2
        return "\n\n".join(parts)

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        section = self._render()
        prompt = PROMPT_TEMPLATE.format(examples_section=section, input=input)
        response = self.call_llm(prompt)
        answer = extract_json_field(response, "final_answer")
        return answer, {
            "full_response": response,
            "num_examples": len(self.examples),
            "num_injected": section.count("\nA: "),
        }

    def get_context_length(self) -> int:
        return len(self._render())

    # ---------------- state ----------------
    def get_state(self) -> str:
        return json.dumps({"examples": self.examples}, indent=2)

    def set_state(self, state: str) -> None:
        data = json.loads(state)
        self.examples = data.get("examples", [])
