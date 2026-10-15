"""Speculative-decode batches never reach the batched GPU MoE's expert bank.

coli_v4_gpu_moe_batch_union (COLI_CUDA_MOE_BATCH=1) serves prefills from a
process-wide expert bank. Its only size gate used to be the prompt-size hint
that generate() sets once per request, so the speculative verify and replay
batches of the decode that follows (target_batch with batch > 1) took the
bank too: re-created after the prefill had released it, and refilled layer
by layer on every draft step. That is silent on output (the CPU union gives
the same bits) and only shows up as VRAM, store traffic and decode speed,
which no fixture without the 4096-wide, 256-expert geometry can reach. So
this pins the shape at source level:

- target_batch passes its use_prefill_pool intent down for the length of the
  call, and the union refuses decode batches before any bank work;
- in generate(), the prefill loop's target_batch passes 1 and every call
  after the prefill's bank release (speculative verify/replay) passes 0.
"""
import re
import unittest
from pathlib import Path

SOURCE = Path(__file__).resolve().parent.parent / "deepseek_v4.c"


def function_body(text, signature):
    start = text.index(signature)
    end = text.index("\n}\n", start)
    return text[start:end]


def generate_body(text):
    """The body that runs generate()'s prefill loop and decode: the public
    coli_v4_session_generate, or the impl it wraps."""
    for signature in ("static int session_generate_impl(",
                      "int coli_v4_session_generate(ColiV4Session"):
        if signature in text:
            body = function_body(text, signature)
            if "target_batch(" in body:
                return body
    raise AssertionError("generate() body not found")


def call_arguments(text, start):
    """Top-level arguments of the call whose '(' is at or after start."""
    open_at = text.index("(", start)
    depth, current, arguments = 0, [], []
    for char in text[open_at + 1:]:
        if char == "(":
            depth += 1
        elif char == ")":
            if depth == 0:
                arguments.append("".join(current).strip())
                return arguments
            depth -= 1
        if char == "," and depth == 0:
            arguments.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    raise AssertionError("unterminated call")


class V4MoeBatchSpeculativeSourceTest(unittest.TestCase):
    def setUp(self):
        self.text = SOURCE.read_text(encoding="utf-8")

    def test_target_batch_passes_the_prefill_intent_down(self):
        body = function_body(self.text, "static int target_batch(ColiV4Engine")
        mark = body.index("coli_v4_gpu_moe_batch_prefill(use_prefill_pool ? 1 : 0);")
        call = body.index("target_batch_impl(", mark)
        clear = body.index("coli_v4_gpu_moe_batch_prefill(-1);", call)
        self.assertLess(mark, call)
        self.assertLess(call, clear)

    def test_union_refuses_decode_batches_before_any_bank_work(self):
        body = function_body(self.text, "int coli_v4_gpu_moe_batch_union(")
        refuse = body.index("if (!v4_gpu_moe_batch_in_prefill) return -1;")
        self.assertLess(refuse, body.index("dsv4_cuda_expert_bank_create("))
        self.assertLess(refuse, body.index("pthread_create(&v4_bank2_thread"))

    def test_generate_prefill_passes_1_and_decode_passes_0(self):
        body = generate_body(self.text)
        release = body.index("coli_v4_gpu_moe_batch_release();")
        calls = [match.start() for match in
                 re.finditer(r"\btarget_batch\(", body)]
        self.assertTrue(any(at < release for at in calls))
        self.assertTrue(any(at > release for at in calls))
        for at in calls:
            arguments = call_arguments(body, at)
            self.assertEqual(len(arguments), 15, arguments)
            self.assertEqual(arguments[10], "1" if at < release else "0",
                             f"target_batch at offset {at}: {arguments}")


if __name__ == "__main__":
    unittest.main()
