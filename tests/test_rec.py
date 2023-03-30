from tomocupy_stream import GPURecRAM
from tomocupy_stream import find_center
import h5py
import numpy as np
import tifffile


# init data, dark, flat, theta
with h5py.File('data/test_data.h5', 'r') as fid:
    data = fid['/exchange/data'][:, :, :]
    dark = fid['/exchange/data_dark'][:]
    flat = fid['/exchange/data_white'][:]
    theta = fid['/exchange/theta'][:]
data = data.swapaxes(0,1)  # note: sinogram shape

center_search_width = 100
center_search_step = 0.5
center_search_ind = data.shape[0]//2
rotation_axis = find_center.find_center_vo(data, dark, flat,
                                           ind=center_search_ind,
                                           smin=-center_search_width, 
                                           smax=center_search_width, 
                                           step=center_search_step)
print('auto rotation axis',rotation_axis)
# create reconstruction class (preallocate memory, init grids for backprojection)
# can be done once
cl = GPURecRAM(
    n=1536,  # sample size in x
    nz=22,  # sample size in z
    nproj=720,  # number of projection angles
    ncz=4,  # chunk size (multiple of 2)
    ndark=10,  # number of dark fields
    nflat=20,  # number of flat fields
    in_dtype='uint16',  # input data type
    dtype='float32',  # computation type, note  for float16 n should be a power of 2
    reconstruction_algorithm='fourierrec',  # fourierrec, lprec, or linerec

    dezinger=0,  # zinger size (0 - no zingers)
    # dezinger = 2,  # removing Zingers
    # dezinger_threshold = 5000,

    remove_stripe_method='None',
    # remove_stripe_method = 'fw',
    # fw_sigma = 1,
    # fw_level = 7,
    # fw_filter = 'sym16',

    # remove_stripe_method = 'ti',
    # ti_beta = 0.022,
    # ti_mask = 1.0,

    fbp_filter='parzen',  # filter for fbp
    rotation_axis=rotation_axis,  # rotation center
)

# run recon
result = cl.recon_all(data, dark, flat, theta)

# save recon
print(np.linalg.norm(result))
tifffile.imwrite('data/result.tiff', result)
