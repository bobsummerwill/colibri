"""The batched GPU MoE's expert bank lives only while a prefill runs.

coli_v4_gpu_moe_batch_union (COLI_CUDA_MOE_BATCH=1) keeps a process-wide
expert bank, and with COLI_CUDA_MOE_DOUBLE a second bank plus a prefetch
worker that holds a store lease. Three things keep that state inside a
prefill, and a regression in any of them is silent on output (the CPU union
gives the same bits) and only shows up as VRAM, disk traffic or a crash at
teardown, which no fixture without the 4096-wide geometry can reach. So this
pins the shape at source level:

- speculative verify/replay batches during decode are refused the GPU union:
  target_batch passes its use_prefill_pool down, the decode calls pass 0;
- every exit of coli_v4_session_generate (cancel and failures included)
  releases the bank, and engine close releases it before teardown;
- a second bank that did not fit is retried by the next prefill, while the
  structural "prefetch produced nothing" latch stays for the process.
"""
import re
import unittest
from pathlib import Path

SOURCE = Path(__file__).resolve().parent.parent / "deepseek_v4.c"


def function_body(text, signature):
    start = text.index(signature)
    end = text.index("\n}\n", start)
    return text[start:end]


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


class V4MoeBatchLifecycleSourceTest(unittest.TestCase):
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
        body = function_body(self.text, "static int session_generate_impl(")
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

    def test_every_generate_exit_releases_the_bank(self):
        body = function_body(self.text, "int coli_v4_session_generate(ColiV4Session")
        call = body.index("int result = session_generate_impl(")
        release = body.index("coli_v4_gpu_moe_batch_release();", call)
        self.assertLess(release, body.index("return result;", release))
        self.assertEqual(body.count("return"), 1)

    def test_engine_close_releases_before_teardown(self):
        body = function_body(self.text, "void coli_v4_gpu_engine_close(")
        release = body.index("coli_v4_gpu_moe_batch_release();")
        for later in ("dsv4_cuda_tensor_free(", "v4_gpu_expert_mirrors_free(",
                      "dsv4_cuda_shutdown();"):
            self.assertLess(release, body.index(later), later)

    def test_only_the_structural_bank2_latch_is_permanent(self):
        release = function_body(self.text, "void coli_v4_gpu_moe_batch_release(")
        self.assertIn("v4_bank2_alloc_failed = 0;", release)
        self.assertNotIn("v4_bank2_disabled = 0;", self.text)
        union = function_body(self.text, "int coli_v4_gpu_moe_batch_union(")
        failed = union.index("if (!v4_moe_bank2) {", union.index(
            "v4_moe_bank2 = dsv4_cuda_expert_bank_create("))
        branch = union[failed:union.index("} else {", failed)]
        self.assertIn("v4_bank2_alloc_failed = 1;", branch)
        self.assertNotIn("v4_bank2_disabled", branch)
        self.assertIn("v4_bank_double_usable() && !v4_bank2_alloc_failed", union)


if __name__ == "__main__":
    unittest.main()
