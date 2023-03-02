from tomocupy_stream import GPURecRAM
import numpy as np
import tifffile
import time


################ a structure of parameters (should be done in a nicer way)
class args:
    pass

args.n = 2048 # sample size in x
args.nz = 2048 # sample size in z
args.nproj = 2048 # number of projection angles
args.ncz = 8 # chunk size (multiple of 2)
args.ndark = 10 # number of dark fields
args.nflat = 10 # number of flat fields
args.in_dtype = 'uint8' # input data type
args.dtype = 'float16' # computation type, note  for float16 n should be a power of 2
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
args.rotation_axis = args.n/2 # rotation center
#####################################################





# init data, dark, flat, theta
data = np.zeros([args.nz,args.nproj,args.n],dtype=args.in_dtype)
data[:,:,args.n//2] = 32
dark = np.zeros([args.ndark,args.nz,args.n],dtype=args.in_dtype)
flat = np.ones([args.nflat,args.nz,args.n],dtype=args.in_dtype)+64
theta = np.linspace(0,180,args.nproj,endpoint=False).astype('float32')

# memory for result
result = np.zeros([args.nz,args.n,args.n],dtype=args.dtype)

# create reconstruction class (preallocate memory, init grids for backprojection)
# can be done once 
cl = GPURecRAM(args, theta)
# run recon
t = time.time()
cl.recon_all(result,data,dark,flat)
print(f'Reconstruction time: {time.time()-t}s')

# save recon
# print(np.linalg.norm(result))
# tifffile.imwrite('data/result.tiff',result)    
