"""Dataset layout and phantom, shared by the server and the client.

    [ proj (nproj, nz, n) | dark (ndark, nz, n) | flat (nflat, nz, n) ]
"""

import zlib

import numpy as np

NDARK = NFLAT = 2
NPLANE = 8  # distinct phantom planes cycled over z
FLAT = 60000.0  # transmitted counts with nothing in the beam


def allocate(nproj, nz, n):
    """One contiguous buffer plus the three views the pipeline wants."""
    nall = (nproj + NDARK + NFLAT) * nz * n
    buf = np.empty(nall, dtype=np.uint16)
    proj = buf[: nproj * nz * n].reshape(nproj, nz, n)
    dark = buf[nproj * nz * n :][: NDARK * nz * n].reshape(NDARK, nz, n)
    flat = buf[(nproj + NDARK) * nz * n :].reshape(NFLAT, nz, n)
    return buf, proj, dark, flat


def theta(nproj):
    return np.linspace(0, np.pi, nproj, endpoint=False).astype("float32")


def _planes(nproj, n):
    """NPLANE sinograms of two analytic disks, as transmitted counts.

    Disk 1 sits at the origin with radius 0.7; disk 2 at (0.25, 0.1) with
    radius 0.2 and an amplitude that varies with the plane index. Both have
    closed-form line integrals, 2*sqrt(r^2 - d^2), so the reconstruction can
    be checked numerically instead of by eye.
    """
    th = np.pi * np.arange(nproj) / nproj
    ct, st = np.cos(th)[:, None], np.sin(th)[:, None]
    s = np.linspace(-1, 1, n)[None, :]

    d1 = 2 * np.sqrt(np.maximum(0.49 - s**2, 0))  # disk 1, same for every angle
    dd = s - (0.25 * ct + 0.1 * st)  # disk 2 shifts with the angle
    chord2 = 2 * np.sqrt(np.maximum(0.04 - dd**2, 0))

    out = np.empty((NPLANE, nproj, n), dtype=np.uint16)
    for k in range(NPLANE):
        a2 = 0.3 + 0.4 * k / NPLANE
        out[k] = np.clip(FLAT * np.exp(-(d1 + a2 * chord2)), 0, 65535)
    return out


def fill(proj, dark, flat):
    """Write the phantom into views returned by allocate()."""
    nproj, nz, n = proj.shape
    planes = _planes(nproj, n)
    # one strided write per plane instead of one per z
    for k in range(NPLANE):
        proj[:, k::NPLANE, :] = planes[k][:, None, :]
    dark[:] = 0
    flat[:] = FLAT


def checksum(buf):
    """Order-sensitive hash, so a reordered or short transfer is caught."""
    return zlib.crc32(memoryview(buf).cast("B")) & 0xFFFFFFFF
