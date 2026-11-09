"""Tiered weight store — the core IP of the workstation.

Model weights live in one packed file on NVMe ("expert-pack" layout: per-layer
expert blocks stored contiguously with a JSON index). A small VRAM cache and a
larger DRAM LRU cache sit in front of it. Hot experts get pinned (learned,
frequency-conditioned) and a prefetcher pulls likely-routed experts into DRAM
ahead of the router's next-token choice.

Tiers: VRAM (hottest, smallest) -> DRAM (LRU) -> NVMe (mmap-backed, full model).
"""
from __future__ import annotations

import json
import mmap
import os
import threading
from collections import Counter, OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass


@dataclass(frozen=True)
class BlockRef:
    layer: int
    expert: int
    offset: int
    nbytes: int


class TieredWeightStore:
    def __init__(self, path: str, dram_bytes: int, vram_bytes: int,
                 pin_fraction: float = 0.1, repin_every: int = 64,
                 repin_min_keys: int = 8):
        self.path = path
        self.dram_bytes = dram_bytes
        self.vram_bytes = vram_bytes
        self.pin_fraction = pin_fraction
        self.repin_every = repin_every
        self.repin_min_keys = repin_min_keys

        self.vram: OrderedDict = OrderedDict()   # key -> bytes (LRU)
        self.dram: OrderedDict = OrderedDict()   # key -> bytes (LRU)
        self.pinned: set = set()
        self.freq: Counter = Counter()
        self._accesses = 0

        self.index: dict = {}
        idx_path = path + ".index.json"
        with open(idx_path) as f:
            raw = json.load(f)
        for k, r in raw.items():
            layer, expert = (int(x) for x in k.split(","))
            self.index[(layer, expert)] = BlockRef(layer, expert, r["offset"], r["nbytes"])

        self._file = open(path, "rb")
        self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="prefetch")

        # stats
        self.hits: Counter = Counter()
        self.nvme_reads = 0
        self.nvme_bytes = 0

    # ------------------------------------------------------------------ pack
    @classmethod
    def pack(cls, path: str, blocks, dram_bytes: int = 0, vram_bytes: int = 0):
        """blocks: iterable of (layer, expert, bytes). Writes packed file + index."""
        index = {}
        with open(path, "wb") as f:
            off = 0
            for layer, expert, data in blocks:
                f.write(data)
                index[f"{layer},{expert}"] = {"offset": off, "nbytes": len(data)}
                off += len(data)
        with open(path + ".index.json", "w") as f:
            json.dump(index, f)
        return cls(path, dram_bytes=dram_bytes, vram_bytes=vram_bytes)

    # ------------------------------------------------------------------ get
    def get(self, layer: int, expert: int) -> bytes:
        key = (layer, expert)
        ref = self.index.get(key)
        if ref is None:
            raise KeyError(f"no block for {key}")
        with self._lock:
            self._accesses += 1
            if self._accesses % self.repin_every == 0:
                self._repin()
            if key in self.vram:
                self.vram.move_to_end(key)
                self.hits["vram"] += 1
                self.freq[key] += 1
                return self.vram[key]
            if key in self.dram:
                self.dram.move_to_end(key)
                data = self.dram[key]
                self.hits["dram"] += 1
                self.freq[key] += 1
                self._promote_vram(key, data)
                return data
            # NVMe tier
            data = self._mmap[ref.offset:ref.offset + ref.nbytes]
            self.hits["nvme"] += 1
            self.nvme_reads += 1
            self.nvme_bytes += ref.nbytes
            self.freq[key] += 1
            self._put_dram(key, data)
            self._promote_vram(key, data)
            return data

    def prefetch(self, layer: int, experts) -> list:
        """Async-pull experts into DRAM (fire-and-forget). Returns futures."""
        return [self._pool.submit(self.get, layer, e) for e in experts]

    # ------------------------------------------------------------- internals
    def _used(self, cache) -> int:
        return sum(len(v) for v in cache.values())

    def _evict(self, cache, cap: int, incoming: int):
        """Evict LRU non-pinned entries until `incoming` bytes fit under `cap`."""
        for k in list(cache.keys()):
            if self._used(cache) + incoming <= cap:
                break
            if k in self.pinned:
                continue
            cache.pop(k)

    def _put_dram(self, key, data: bytes):
        if key in self.dram:
            self.dram.move_to_end(key)
            return
        if len(data) > self.dram_bytes:
            return  # too big for DRAM tier; NVMe-only
        self._evict(self.dram, self.dram_bytes, len(data))
        self.dram[key] = data
        self.dram.move_to_end(key)

    def _promote_vram(self, key, data: bytes):
        if key in self.vram:
            self.vram.move_to_end(key)
            return
        if len(data) > self.vram_bytes:
            return
        self._evict(self.vram, self.vram_bytes, len(data))
        self.vram[key] = data
        self.vram.move_to_end(key)

    def _repin(self):
        if len(self.freq) < self.repin_min_keys:
            return
        k = max(1, int(self.pin_fraction * len(self.freq)))
        self.pinned = set(k for k, _ in self.freq.most_common(k))
        # pinned experts must be DRAM-resident
        for key in self.pinned:
            if key not in self.dram:
                ref = self.index[key]
                self._put_dram(key, self._mmap[ref.offset:ref.offset + ref.nbytes])

    # ---------------------------------------------------------------- stats
    def stats(self) -> dict:
        total = sum(self.hits.values()) or 1
        return {
            "hits": {t: self.hits.get(t, 0) for t in ("vram", "dram", "nvme")},
            "hit_rate": {t: round(self.hits[t] / total, 3) for t in ("vram", "dram", "nvme")},
            "nvme_reads": self.nvme_reads,
            "nvme_bytes": self.nvme_bytes,
            "dram_resident": len(self.dram),
            "vram_resident": len(self.vram),
            "pinned": sorted(self.pinned),
        }

    def close(self):
        self._pool.shutdown(wait=False)
        self._mmap.close()
        self._file.close()
