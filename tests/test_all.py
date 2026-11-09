"""Stdlib-only tests: run with `python -m unittest discover -s tests -v`."""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from weights.store import TieredWeightStore          # noqa: E402
from runtime import (MemoryPool, Stream, Tier,        # noqa: E402
                     list_devices, get_device, register_kernel)

BLOCK = 1024  # 1 KB blocks for tests


class TestWeightStore(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.path = os.path.join(self.tmp, "model.bin")
        # 3 layers x 3 experts
        blocks = [(l, e, bytes([l * 10 + e]) * BLOCK) for l in range(3) for e in range(3)]
        # DRAM fits 4 blocks, VRAM fits 2
        self.store = TieredWeightStore.pack(self.path, blocks, dram_bytes=4 * BLOCK,
                                            vram_bytes=2 * BLOCK)

    def tearDown(self):
        self.store.close()

    def test_first_sweep_reads_all_from_nvme(self):
        for l in range(3):
            for e in range(3):
                data = self.store.get(l, e)
                self.assertEqual(len(data), BLOCK)
        self.assertEqual(self.store.hits["nvme"], 9)
        self.assertEqual(self.store.nvme_reads, 9)

    def test_cache_hits_on_repeat_access(self):
        for l in range(3):
            for e in range(3):
                self.store.get(l, e)
        # (1,2) and (2,0) are DRAM-resident but evicted from VRAM after the sweep
        self.store.get(1, 2)
        self.store.get(2, 0)   # -> DRAM hits, promoted to VRAM
        self.store.get(1, 2)
        self.store.get(2, 0)   # -> VRAM hits
        s = self.store.stats()
        self.assertEqual(s["hits"]["nvme"], 9)
        self.assertEqual(s["hits"]["dram"], 2)
        self.assertEqual(s["hits"]["vram"], 2)

    def test_full_sweep_thrashes_lru(self):
        # classic LRU pathology: sweeping a set larger than the cache = 0% hits
        for _ in range(2):
            for l in range(3):
                for e in range(3):
                    self.store.get(l, e)
        s = self.store.stats()
        self.assertEqual(s["hits"]["nvme"], 18)
        self.assertEqual(s["hits"]["dram"] + s["hits"]["vram"], 0)

    def test_hot_expert_gets_pinned(self):
        hot = (0, 0)
        for i in range(200):
            self.store.get(*hot)
            # churn other experts so LRU would normally evict things
            for l in range(3):
                for e in range(3):
                    if (l, e) != hot:
                        self.store.get(l, e)
        self.assertIn(hot, self.store.pinned)
        # the pinned hot expert must never be re-read from NVMe once pinned
        before = self.store.hits["nvme"]
        for _ in range(50):
            self.store.get(*hot)
        self.assertEqual(self.store.hits["nvme"], before)

    def test_prefetch_warms_dram(self):
        futs = self.store.prefetch(1, [0, 1, 2])
        for f in futs:
            f.result()
        before = self.store.hits["nvme"]
        for e in range(3):
            self.store.get(1, e)
        self.assertEqual(self.store.hits["nvme"], before)  # no new NVMe reads


class TestRuntimeAPI(unittest.TestCase):
    def test_devices_enumerate(self):
        devs = list_devices()
        kinds = {d.kind.value for d in devs}
        self.assertIn("gpu", kinds)
        gpu = get_device("gpu0")
        self.assertEqual(gpu.kind.value, "gpu")

    def test_pool_tiers_fill_in_order(self):
        pool = MemoryPool(vram_bytes=100, dram_bytes=200)
        t1 = pool.allocate(60)          # fits VRAM
        t2 = pool.allocate(60)          # VRAM full -> DRAM
        t3 = pool.allocate(200)         # DRAM full -> NVMe
        self.assertEqual(t1.tier, Tier.VRAM)
        self.assertEqual(t2.tier, Tier.DRAM)
        self.assertEqual(t3.tier, Tier.NVME)
        t1.free()
        t4 = pool.allocate(60)          # freed VRAM space reused
        self.assertEqual(t4.tier, Tier.VRAM)

    def test_stream_runs_registered_kernel_in_order(self):
        calls = []

        @register_kernel("test_add")
        def _(a, b):
            calls.append((a, b))

        gpu = get_device("gpu0")
        s = Stream(gpu)
        s.submit("test_add", 1, 2)
        s.submit("test_add", 3, 4)
        s.synchronize()
        self.assertEqual(calls, [(1, 2), (3, 4)])
        self.assertEqual(len(s), 0)


if __name__ == "__main__":
    unittest.main()
