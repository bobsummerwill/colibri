"""Staged uploads straight from host memory (up_import): pipelined, and measured before they are kept.

dev recorded every import of an upload into the slot being filled and submitted it only at
up_finish (up_add_host never counted its bytes), so a group of experts (st_fill_group's,
up to 192 matrices) pinned every page first and copied after: no import overlapped a copy.
And on NVIDIA the pin and the unpin cost ~0.4 ms an import, more than the copy into the
staging buffer they save: an RTX 3090 moved 6.6 GB/s imported against 13.1 staged, an RTX
5070 6.7 against 11.6, while an Intel UHD 770 imports for ~0.002 ms and gains (8.8 against
7.4 with the pipelining). So the slot is submitted once its copies reach VK_UP_SLOT bytes,
and with COLI_VK_UP_IMPORT unset the imports are kept only where a short measurement at
start finds them faster. A real run needs the device, so this checks the source.
"""
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def body(src, sig):
    start = src.index(sig)
    return src[start:src.index("\n}\n", start)]


class VkUpImport(unittest.TestCase):
    def setUp(self):
        self.src = (ROOT / "backend_vulkan.c").read_text(encoding="utf-8")

    def test_imports_submit_the_slot(self):
        f = body(self.src, "static int up_add_host(VkUp *u, int dev,")
        self.assertIn("u->imp_used += len[i];", f)
        self.assertIn("if (u->used + u->imp_used >= VK_UP_SLOT) up_submit(u, dev);", f)
        self.assertIn("u->open = 1; u->used = 0; u->imp_used = 0;", self.src)

    def test_unset_is_measured(self):
        f = body(self.src, "static void up_import_init(VkUp *u, int dev,")
        self.assertIn("if (!(e && *e) && !coli_vk_fault_set()) {", f)
        self.assertIn("int r = up_import_probe(u, dev, ms);", f)
        # staged faster: no imports on this device
        self.assertIn("u->imp_align = 0; u->imp_props = NULL;\n            return;", f)

    def test_probe_leaves_the_report_alone(self):
        f = body(self.src, "static int up_import_probe(VkUp *u, int dev, double ms[2]) {\n    enum")
        self.assertIn("u->imp_bytes = keep[3]; u->imp_copies = keep[4];", f)
        self.assertIn("pthread_mutex_lock(&u->mx);", f)
        self.assertIn("return ms[1] < ms[0];", f)

    def test_both_devices_measure(self):
        self.assertIn('up_import_init(&g_up[0], 0, G.phys, G.has_hostmem, "");', self.src)
        self.assertIn('up_import_init(u, 1, G2.phys, G2.has_hostmem, "dev2 ");', self.src)


if __name__ == "__main__":
    unittest.main()
