"""The batched GPU MoE's expert bank lives only while a prefill runs.

coli_v4_gpu_moe_batch_union (COLI_CUDA_MOE_BATCH=1) keeps a process-wide
expert bank, and with COLI_CUDA_MOE_DOUBLE a second bank plus a prefetch
worker that holds a store lease. A regression in releasing that state is
silent on output (the CPU union gives the same bits) and only shows up as
VRAM, disk traffic or a crash at teardown, which no fixture without the
4096-wide geometry can reach. So this pins the shape at source level:

- every exit of coli_v4_session_generate (cancel and failures included)
  releases the bank, and engine close releases it before teardown.
"""
import unittest
from pathlib import Path

SOURCE = Path(__file__).resolve().parent.parent / "deepseek_v4.c"


def function_body(text, signature):
    start = text.index(signature)
    end = text.index("\n}\n", start)
    return text[start:end]


class V4MoeBatchLifecycleSourceTest(unittest.TestCase):
    def setUp(self):
        self.text = SOURCE.read_text(encoding="utf-8")

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


if __name__ == "__main__":
    unittest.main()
