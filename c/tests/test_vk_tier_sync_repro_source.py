"""COLI_VK_TIER_SYNC=1 keeps the streaming rule unmeasured, so the documented reproducible run repeats.

The docs promise that COLI_VK_TIER_SYNC=1 COLI_VK_TIER_BALANCE=0 make a run with the tier
reproducible. But the rule that streams a big prompt step's cold experts to the device,
R = max(G, ceil(upload / cpu_row)), took the upload rate and the CPU's time per row as
measured, so R (16 in one run of V4 Flash REAP on an RTX 3090, 23 in another) and with it
the device/CPU split, the expert sums' bits and in the end a greedy text moved from run to
run. Under SYNC the rule keeps R = G. A real run needs the device, so this checks the source.
"""
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def body(src, sig):
    start = src.index(sig)
    return src[start:src.index("\n}\n", start)]


class VkTierSyncRepro(unittest.TestCase):
    def setUp(self):
        self.src = (ROOT / "vk_tier.c").read_text(encoding="utf-8")

    def test_sync_takes_no_measured_time(self):
        f = body(self.src, "static int stream_min_rows(void) {")
        self.assertIn("if (T.st_rows_env > 0) return T.st_rows_env;", f)
        self.assertIn("if (!T.sync && T.bw_up > 0 && T.cpu_row_ms > 0) {", f)

    def test_the_line_says_so(self):
        self.assertIn("COLI_VK_TIER_SYNC takes no measured time", self.src)

    def test_documented(self):
        env = (ROOT.parent / "docs" / "ENVIRONMENT.md").read_text(encoding="utf-8")
        self.assertIn("the streaming rule takes no measured time", env)


if __name__ == "__main__":
    unittest.main()
