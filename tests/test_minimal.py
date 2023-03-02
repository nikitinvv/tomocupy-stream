from tomocupy_stream import GPURecRAM
import numpy as np
import tifffile
import time


in_dtype = "uint8"  # input data type
dtype = "float32"  # output data type
n = 2048  # sample size in x
nz = 2048  # sample size in z
nproj = 2048  # number of projection angles
ndark = 10  # number of dark fields
nflat = 10  # number of flat fields

# init data, dark, flat, theta
data = np.zeros([nz, nproj, n], dtype=in_dtype)
data[:, :, n // 2] = 32
dark = np.zeros([ndark, nz, n], dtype=in_dtype)
flat = np.ones([nflat, nz, n], dtype=in_dtype) + 64
theta = np.linspace(0, 180, nproj, endpoint=False).astype("float32")


cl = GPURecRAM.for_data_like(
    data=data,
    dark=dark,
    flat=flat,
    ncz=8,  # chunk size (multiple of 2)
    rotation_axis=2048 / 2,  # rotation center
    dtype="float32",  # computation type, note  for float16 n should be a power of 2
)
t = time.time()
result = cl.recon_all(data, dark, flat, theta)
print(f"Reconstruction time: {time.time()-t}s")
