"""Build a tomography dataset in this host's RAM

    python tomo_server.py [nproj nz n]
"""

import sys
import time

import rdma
import tomo_data


def main():
    nproj, nz, n = (int(a) for a in (sys.argv[1:4] or [1024, 512, 2048]))

    buf, proj, dark, flat = tomo_data.allocate(nproj, nz, n)
    print(f"dataset {nproj} x {nz} x {n} = {buf.nbytes / 1e9:.2f} GB in RAM")

    t = time.time()
    tomo_data.fill(proj, dark, flat)
    print(f"phantom built in {time.time() - t:.2f} s")
    print(f"checksum {tomo_data.checksum(buf):08x}")

    # blocks until the client has pulled everything
    rdma.serve(buf)
    print("client finished")


if __name__ == "__main__":
    main()
