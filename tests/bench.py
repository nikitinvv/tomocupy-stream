"""Synthetic end-to-end benchmark, one configuration per process.

    python bench.py --n 2048 --mode rec --ngpus 4 --dtype float16

Prints one RESULT line, so runs can be compared across builds.
"""

import argparse
import json
import resource
import threading
import time

import numpy as np

from streamtomocupy import config
from streamtomocupy import streamrecon

NDARK = NFLAT = 2
NPLANE = 8  # distinct phantom planes cycled over z
FLAT = 60000.0  # transmitted counts with nothing in the beam


def get_data_pars(args, proj, flat, dark):
    args.nproj = proj.shape[0]
    args.nz = proj.shape[1]
    args.n = proj.shape[2]
    args.nflat = flat.shape[0]
    args.ndark = dark.shape[0]
    args.in_dtype = proj.dtype
    return args


def _planes(nproj, n):
    """NPLANE sinograms of two analytic disks, as transmitted counts."""
    th = np.pi * np.arange(nproj) / nproj
    ct, st = np.cos(th)[:, None], np.sin(th)[:, None]
    s = np.linspace(-1, 1, n)[None, :]

    d1 = 2 * np.sqrt(np.maximum(0.49 - s**2, 0))  # disk 1, angle independent
    dd = s - (0.25 * ct + 0.1 * st)  # disk 2 shifts with the angle
    chord2 = 2 * np.sqrt(np.maximum(0.04 - dd**2, 0))

    out = np.empty((NPLANE, nproj, n), dtype=np.uint16)
    for k in range(NPLANE):
        out[k] = np.clip(FLAT * np.exp(-(d1 + (0.3 + 0.4 * k / NPLANE) * chord2)),
                         0, 65535)
    return out


def make_data(nproj, nz, n, phantom):
    """Allocate and fill the input. np.empty first, so nothing is written twice."""
    dark = np.zeros((NDARK, nz, n), dtype=np.uint16)
    flat = np.full((NFLAT, nz, n), int(FLAT), dtype=np.uint16)
    proj = np.empty((nproj, nz, n), dtype=np.uint16)
    if phantom:
        planes = _planes(nproj, n)
        for k in range(NPLANE):  # one strided write per plane
            proj[:, k::NPLANE, :] = planes[k][:, None, :]
    else:
        proj[:] = 100
    theta = np.linspace(0, np.pi, nproj, endpoint=False).astype("float32")
    return proj, dark, flat, theta


def chunked_norm(a, step=32):
    """Norm in float32 without copying the whole volume."""
    total = 0.0
    for i in range(0, a.shape[0], step):
        b = a[i : i + step].astype(np.float32).ravel()
        total += float(np.dot(b, b))
    return total**0.5


def peak_rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2


class GpuWatch:
    """Samples used memory on device 0, including allocations outside cupy's pool."""

    def __init__(self, period=0.2):
        self.period = period
        self.peak = 0
        self._stop = threading.Event()
        self._th = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        import cupy as cp

        with cp.cuda.Device(0):
            while not self._stop.wait(self.period):
                free, total = cp.cuda.runtime.memGetInfo()
                self.peak = max(self.peak, total - free)

    def __enter__(self):
        self._th.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._th.join(timeout=2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--conf", default="test.conf")
    p.add_argument("--n", type=int, default=2048, help="nproj = nz = n")
    p.add_argument("--nproj", type=int, default=0, help="override nproj")
    p.add_argument("--nz", type=int, default=0, help="override nz, to probe gpu memory")
    p.add_argument("--mode", choices=["rec", "rec_steps"], default="rec")
    p.add_argument("--algorithm", default="fourierrec")
    p.add_argument("--dtype", default="float32")
    p.add_argument("--ngpus", type=int, default=1)
    p.add_argument("--nsino-per-chunk", type=int, default=8)
    p.add_argument("--nproj-per-chunk", type=int, default=8)
    p.add_argument("--phantom", action="store_true", help="analytic disks, not a constant")
    p.add_argument("--norm", action="store_true", help="also report the result norm")
    p.add_argument("--tag", default="")
    a = p.parse_args()

    n = a.n
    nproj = a.nproj or n
    nz = a.nz or n

    t = time.time()
    proj, dark, flat, theta = make_data(nproj, nz, n, a.phantom)
    t_data = time.time() - t

    args = config.read_args(a.conf)
    args = get_data_pars(args, proj, flat, dark)
    args.reconstruction_algorithm = a.algorithm
    args.dtype = a.dtype
    args.ngpus = a.ngpus
    args.nsino_per_chunk = a.nsino_per_chunk
    args.nproj_per_chunk = a.nproj_per_chunk

    with GpuWatch() as gw:
        t = time.time()
        cl = streamrecon.StreamRecon(args)
        t_class = time.time() - t

        t = time.time()
        getattr(cl, a.mode)(proj, dark, flat, theta)
        t_rec = time.time() - t

    out = {
        "tag": a.tag,
        "n": n,
        "nproj": nproj,
        "nz": nz,
        "mode": a.mode,
        "algorithm": a.algorithm,
        "dtype": a.dtype,
        "ngpus": a.ngpus,
        "ncz": a.nsino_per_chunk,
        "ncproj": a.nproj_per_chunk,
        "t_data": round(t_data, 2),
        "t_class": round(t_class, 2),
        "t_rec": round(t_rec, 2),
        "peak_rss_gb": round(peak_rss_gb(), 1),
        "peak_gpu_gb": round(gw.peak / 1024**3, 1),
    }
    if a.norm:
        out["norm"] = round(chunked_norm(cl.get_res()[2]), 4)
    print("RESULT " + json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
