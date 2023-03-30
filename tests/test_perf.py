from tomocupy_stream import GPURecRAM
from tomocupy_stream import find_center
import numpy as np
import tifffile
import time


in_dtype = 'uint8'  # input data type
dtype = 'float32'  # output data type
n = 2048  # sample size in x
nz = 2048  # sample size in z
nproj = 2048  # number of projection angles
ndark = 10  # number of dark fields
nflat = 10  # number of flat fields

# init data, dark, flat, theta
data = np.zeros([nz,nproj,n],dtype=in_dtype)
data[:,:,n//2] = 32
dark = np.zeros([ndark,nz,n],dtype=in_dtype)
flat = np.ones([nflat,nz,n],dtype=in_dtype)+64
theta = np.linspace(0,180,nproj,endpoint=False).astype('float32')

# memory for result
result = np.zeros([nz,n,n],dtype=dtype)


t = time.time()
center_search_width = 100
center_search_step = 0.5
center_search_ind = data.shape[0]//2
rotation_axis = find_center.find_center_vo(data, dark, flat,
                                           ind=center_search_ind,
                                           smin=-center_search_width, 
                                           smax=center_search_width, 
                                           step=center_search_step)
print(f'Center search time: {time.time()-t}s')
print('auto rotation axis',rotation_axis)

# create reconstruction class (preallocate memory, init grids for backprojection)
# can be done once 
cl = GPURecRAM(
    n = n,
    nz = nz,
    nproj = nproj,
    ncz = 8,  # chunk size (multiple of 2)
    ndark = ndark,
    nflat = nflat,
    in_dtype = in_dtype,
    dtype = 'float32',  # computation type, note  for float16 n should be a power of 2
    reconstruction_algorithm = 'fourierrec',  # fourierrec, lprec, or linerec

    dezinger = 0, # zinger size (0 - no zingers)
    # dezinger = 2, # removing Zingers
    # dezinger_threshold = 5000,

    remove_stripe_method = 'None',
    # remove_stripe_method = 'fw',
    # fw_sigma = 1,
    # fw_level = 7,
    # fw_filter = 'sym16',

    # remove_stripe_method = 'ti',
    # ti_beta = 0.022,
    # ti_mask = 1.0,

    fbp_filter = 'parzen',  # filter for fbp
    rotation_axis = rotation_axis, # rotation center
)
# run recon
t = time.time()
cl.recon_all(data, dark, flat, theta, output=result)
print(f'Reconstruction time: {time.time()-t}s')

# save recon
# print(np.linalg.norm(result))
# tifffile.imwrite('data/result.tiff',result)    
