"""Thin ctypes wrapper over rdma_transport.c.

    rdma.serve(arr)                 # expose arr, block until the peer has it
    secs = rdma.fetch(ip, arr)      # pull into arr, return the transfer time
"""

import ctypes
import os
import socket
import subprocess
import time

HERE = os.path.dirname(os.path.abspath(__file__))
# the home directory is shared between the hosts, so keep one .so per host
LIB = os.path.join(HERE, f"librdma.{socket.gethostname().split('.')[0]}.so")
SRC = os.path.join(HERE, "rdma_transport.c")
PORT = 18515


def _load():
    """Compile the transport if the .so is stale, then load it."""
    if not os.path.exists(LIB) or os.path.getmtime(LIB) < os.path.getmtime(SRC):
        subprocess.run(
            ["gcc", "-O2", "-fPIC", "-shared", "-Wall", "-o", LIB, SRC,
             "-lrdmacm", "-libverbs"],
            check=True,
        )
    lib = ctypes.CDLL(LIB)
    lib.rdma_serve.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    lib.rdma_fetch.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_void_p,
                               ctypes.c_size_t, ctypes.POINTER(ctypes.c_double)]
    lib.rdma_errmsg.restype = ctypes.c_char_p
    return lib


_lib = _load()


def _buffer(arr):
    if not arr.flags["C_CONTIGUOUS"]:
        raise ValueError("array must be C-contiguous; the HCA reads it as is")
    return arr.ctypes.data, arr.nbytes


def serve(arr, port=PORT):
    """Expose arr for remote reads and block until the peer has pulled it."""
    ptr, nbytes = _buffer(arr)
    # ctypes drops the GIL around the call, so the rest of the process is free
    if _lib.rdma_serve(ptr, nbytes, port) != 0:
        raise RuntimeError(_lib.rdma_errmsg().decode())


# a peer that is not listening yet fails the connect in one of these ways
_NOT_UP = ("REJECTED", "UNREACHABLE", "ADDR_ERROR", "CONNECT_ERROR",
           "rdma_resolve_addr", "rdma_resolve_route")


def fetch(ip, arr, port=PORT, wait=120):
    """Pull into arr from the server at ip. Returns seconds elapsed.

    Retries while the peer is still coming up, so either side can start first.
    """
    ptr, nbytes = _buffer(arr)
    secs = ctypes.c_double()
    deadline = time.time() + wait
    while True:
        if _lib.rdma_fetch(ip.encode(), port, ptr, nbytes,
                           ctypes.byref(secs)) == 0:
            return secs.value
        err = _lib.rdma_errmsg().decode()
        if time.time() >= deadline or not any(s in err for s in _NOT_UP):
            raise RuntimeError(err)
        time.sleep(1)
