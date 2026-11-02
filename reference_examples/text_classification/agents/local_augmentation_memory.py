"""Local Augmentation Memory.

Mechanism (new relative to every stored / whole-exemplar memory): the memory
stores *answer-level edit operations* rather than demonstrations.

  - Learning keeps every solved (problem, answer) pair.  Additionally, for each
    solved problem it compares the problem text with its own answer and derives
    an *edit tuple*: which pieces of the answer are simply COPIES of the problem
    (retrieved verbatim) and which pieces are ADDED (new text not present in the
    problem).  That tuple is stored next to the example.
  - Prediction retrieves the k nearest solved problems (using the calibrated
    similarity from `_tokens`) and renders them as AUGMENTATION RULES:

        RULE: if the problem says  <verbatim fragment of demo problem>
              then answer = <that fragment> + <added text>

    i.e. the prompt teaches the *transformation* (what is kept, what is
    introduced, what is removed) instead of asking the solver to imitate a flat
    input/output list.  A tiny number of raw demos still anchors the output
    format.

Rationale: when answers are long payloads that are mostly copies of the input
(structured editing tasks: retrosynthesis, normalisation, extraction), a flat
demo bank forces the solver to re-derive the copy structure by itself and burns
context on redundant text.  Explicitly storing "input -> added fragment" makes
the copy mechanics visible with very little context.

General-purpose: no dataset-specific knowledge; the rule mining only relies on
character-level overlap between the problem text and the answer.
"""

import difflib
import json
import re
from typing import Any

from ..llm import LLMCallable
from ..memory_system import MemorySystem, extract_json_field

PROMPT_TEMPLATE = """Solve the problem below. The AUGMENTATION RULES show solved cases as
operations: which part of the answer is copied verbatim from the problem and which
part is newly added. Use them to figure out the transformation, then apply it.

{rules_section}

**Problem:**
{input}

**Instructions:**
- Follow the answer conventions of the anchoring examples
- Respond in JSON format

{{"reasoning": "[brief reasoning]", "final_answer": "[your answer]"}}"""

MAX_RULES = 10
MAX_ANCHORS = 2
CTX_BUDGET = 14000
SNIP = 300
FRAG = 110

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


def _edits(question: str, answer: str) -> tuple[list[str], list[str]]:
    """Split the answer into pieces copied from the question vs newly added."""
    kept: list[str] = []
    added: list[str] = []
    sm = difflib.SequenceMatcher(None, question, answer, autojunk=False)
    for tag, _i1, _i2, j1, j2 in sm.get_opcodes():
        piece = answer[j1:j2]
        if len(piece.strip()) < 2:
            continue
        if tag == "equal":
            kept.append(piece)
        else:
            added.append(piece)
    return kept, added


class LocalAugmentationMemory(MemorySystem):
    """Memory of copy/add edit operations mined from solved examples."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self.examples: list[dict[str, Any]] = []

    # ---------------- learning ----------------
    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        for r in batch_results:
            q = " ".join(str(r.get("raw_question") or r["input"]).split())
            a = str(r["ground_truth"]).strip()
            if not a:
                continue
            # the "problem text" for edit mining = the compacted question
            compact_q = q[:SNIP]
            kept, added = _edits(compact_q, a)
            self.examples.append(
                {
                    "q": compact_q,
                    "a": a,
                    "kept": kept[:3],
                    "added": added[:3],
                    "tok": list(_tokens(compact_q)),
                }
            )

    # ---------------- retrieval ----------------
    def _render(self, input: str) -> str:
        if not self.examples:
            return ""
        qtok = _tokens(input)
        scored = sorted(
            self.examples,
            key=lambda ex: -_sim(qtok, set(ex["tok"])),
        )
        parts: list[str] = []
        total = 0
        for ex in scored[:MAX_RULES]:
            if ex["kept"] and ex["added"]:
                rule = (
                    "RULE: copy-verbatim part(s) "
                    + " / ".join(f'"{_clip(k, FRAG)}"' for k in ex["kept"])
                    + "  ==>  keep them, and write the answer as: "
                    + " ".join(f"+ {_clip(a, FRAG)}" for a in ex["added"])
                    + f"\n  (solved problem: {_clip(ex['q'], 160)})"
                )
            elif ex["added"]:
                rule = (
                    "RULE: nothing in the problem is reused; write: "
                    + " ".join(f"+ {_clip(a, FRAG)}" for a in ex["added"])
                    + f"\n  (solved problem: {_clip(ex['q'], 160)})"
                )
            else:
                rule = (
                    "RULE: reuse only "
                    + " / ".join(f'"{_clip(k, FRAG)}"' for k in ex["kept"])
                    + f"\n  (solved problem: {_clip(ex['q'], 160)})"
                )
            if total + len(rule) > CTX_BUDGET and parts:
                break
            parts.append(rule)
            total += len(rule) + 2

        anchors = []
        for ex in scored[:MAX_ANCHORS]:
            blk = f"Q: {ex['q']}\nA: {ex['a']}"
            if total + len(blk) > CTX_BUDGET * 1.3:
                break
            anchors.append(blk)
            total += len(blk) + 2

        out = ""
        if parts:
            out += "AUGMENTATION RULES (copy/add operations):\n\n" + "\n\n".join(parts)
        if anchors:
            out += "\n\nANCHOR EXAMPLES (full):\n\n" + "\n\n".join(anchors)
        return out

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        section = self._render(input)
        prompt = PROMPT_TEMPLATE.format(rules_section=section, input=input)
        response = self.call_llm(prompt)
        answer = extract_json_field(response, "final_answer")
        return answer, {
            "full_response": response,
            "num_examples": len(self.examples),
        }

    def get_context_length(self) -> int:
        if not self.examples:
            return 0
        return len(self._render(self.examples[len(self.examples) // 2]["q"]))

    # ---------------- state ----------------
    def get_state(self) -> str:
        return json.dumps({"examples": self.examples}, indent=2, ensure_ascii=False)

    def set_state(self, state: str) -> None:
        data = json.loads(state)
        self.examples = data.get("examples", [])
        for ex in self.examples:
            if "tok" not in ex:
                ex["tok"] = list(_tokens(ex["q"]))
