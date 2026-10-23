"""Pull the dataset out of the server's RAM and reconstruct it here.


    python tomo_client.py <server-ip> [nproj nz n]
"""

import os
import sys

import numpy as np

import rdma
import tomo_data
from streamtomocupy import config
from streamtomocupy import streamrecon

HERE = os.path.dirname(os.path.abspath(__file__))
CONF = os.path.join(HERE, "rdma.conf")


def get_data_pars(args, proj, flat, dark):
    """Get parameters of the data"""

    args.nproj = proj.shape[0]
    args.nz = proj.shape[1]
    args.n = proj.shape[2]
    args.nflat = flat.shape[0]
    args.ndark = dark.shape[0]
    args.in_dtype = proj.dtype
    return args


def reconstruct(proj, dark, flat, theta):
    args = config.read_args(CONF)
    args = get_data_pars(args, proj, flat, dark)
    args.reconstruction_algorithm = "fourierrec"
    args.dtype = "float32"
    args.nsino_per_chunk = 8
    args.nproj_per_chunk = 8

    cl = streamrecon.StreamRecon(args)
    cl.rec(proj, dark, flat, theta)
    return cl.get_res()[2]  # res[2] is [nz, n, n]; res[0]/res[1] are sinograms


def write_png(img, path):
    """Save the slice as 8-bit grayscale, scaled 0..max."""
    import matplotlib

    matplotlib.use("Agg")  # no display on the compute nodes
    import matplotlib.pyplot as plt

    # the brightest thing in the slice is disk2's plateau, so scaling to the
    # max keeps both disks distinguishable instead of saturating the small one
    plt.imsave(path, img, cmap="gray", vmin=0, vmax=img.max())
    print(f"wrote {path}")


def main():
    ip = sys.argv[1]
    nproj, nz, n = (int(a) for a in (sys.argv[2:5] or [1024, 512, 2048]))

    buf, proj, dark, flat = tomo_data.allocate(nproj, nz, n)
    print(f"pulling {buf.nbytes / 1e9:.2f} GB from {ip}")

    secs = rdma.fetch(ip, buf)
    print(f"RDMA READ: {buf.nbytes / 1e9:.2f} GB in {secs:.2f} s "
          f"= {buf.nbytes / 1e9 / secs:.2f} GB/s")
    print(f"checksum {tomo_data.checksum(buf):08x}")

    rec = reconstruct(proj, dark, flat, tomo_data.theta(nproj))
    print(f"reconstruction {rec.shape} {rec.dtype}  "
          f"norm {np.linalg.norm(rec):.4f}")

    mid = rec[nz // 2]
    np.save(os.path.join(HERE, "rec_slice.npy"), mid)
    write_png(mid, os.path.join(HERE, "rec_slice.png"))


if __name__ == "__main__":
    main()
