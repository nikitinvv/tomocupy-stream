from tomocupy_stream import GPURecRAM
import h5py
import numpy as np
import tifffile



################ a structure of parameters (should be done in a nicer way)
class args:
    pass

args.n = 1536 # sample size in x
args.nz = 22 # sample size in z
args.nproj = 720 # number of projection angles
args.ncz = 4 # chunk size (multiple of 2)
args.ndark = 10 # number of dark fields
args.nflat = 20 # number of flat fields
args.in_dtype = 'uint16' # input data type
args.dtype = 'float32' # computation type, note  for float16 n should be a power of 2
args.reconstruction_algorithm = 'fourierrec' # fourierrec, lprec, or linerec

args.dezinger = 0 # zinger size (0 - no zingers)
# args.dezinger = 2 # removing Zingers
# args.dezinger_threshold = 5000

args.remove_stripe_method = 'None'
# args.remove_stripe_method = 'fw'
# args.fw_sigma = 1
# args.fw_level = 7
# args.fw_filter = 'sym16'

# args.remove_stripe_method = 'ti'
# args.ti_beta = 0.022
# args.ti_mask = 1.0

args.fbp_filter = 'parzen' # filter for fbp
args.rotation_axis = 782.5 # rotation center
#####################################################





# init data, dark, flat, theta
with h5py.File('data/test_data.h5','r') as fid:        
    data = fid['/exchange/data'][:,:,:]
    dark = fid['/exchange/data_dark'][:]
    flat = fid['/exchange/data_white'][:]
    theta = fid['/exchange/theta'][:]
data = data.swapaxes(0,1)# note: sinogram shape
# memory for result
result = np.zeros([args.nz,args.n,args.n],dtype=args.dtype)

# create reconstruction class (preallocate memory, init grids for backprojection)
# can be done once 
cl = GPURecRAM(args, theta)

# run recon
cl.recon_all(result,data,dark,flat)

# save recon
print(np.linalg.norm(result))
tifffile.imwrite('data/result.tiff',result)    
