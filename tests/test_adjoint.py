from tomocupy_stream import GPURecRAM
import numpy as np
import tifffile
import dxchange


theta = np.linspace(0,180,256).astype('float32')
obj = np.random.random([256,256,256]).astype('float32')*1e-5#[:,64:-64,64:-64]
dark = np.zeros([1,obj.shape[0],obj.shape[-1]],dtype='float32')
flat = np.ones([1,obj.shape[0],obj.shape[-1]],dtype='float32')
data = np.zeros([len(theta),obj.shape[0],obj.shape[-1]],dtype='float32')
data = data.swapaxes(0,1)
cl = GPURecRAM.for_data_like(
    data=data,
    dark=dark,
    flat=flat,
    ncz=8,  # chunk size for GPU processing (multiple of 2), 
    rotation_axis=data.shape[-1]/2,  # rotation center
    dtype="float32",  # computation type, note  for float16 n should be a power of 2
    fbp_filter='none',
    minus_log=False
)
data_reproj = cl.proj_all(obj,theta)
obj_reproj = cl.recon_all(data_reproj, dark, flat, theta)
print('Adjoint test:')
print(f'{np.sum(data_reproj*data_reproj)} ? {np.sum(obj*obj_reproj)}')


# cl = GPURecRAM.for_data_like(
#     data=data,
#     dark=dark,
#     flat=flat,
#     ncz=8,  # chunk size for GPU processing (multiple of 2), 
#     rotation_axis=data.shape[-1]/2,  # rotation center
#     dtype="float32",  # computation type, note  for float16 n should be a power of 2
#     fbp_filter='ramp',
#     minus_log=False
# )
# c = len(theta)/obj.shape[-2]/obj.shape[-2]*np.pi*8
# obj_reproj = cl.recon_all(data_reproj, dark, flat, theta)*c

# print(np.linalg.norm(obj))
# print(np.linalg.norm(obj_reproj))
# print(np.linalg.norm(obj_reproj)/np.linalg.norm(obj))
# dxchange.write_tiff(data_reproj,'res/data_reproj.tiff',overwrite=True)
# dxchange.write_tiff(obj,'res/obj.tiff',overwrite=True)
# dxchange.write_tiff(obj_reproj,'res/objrec.tiff',overwrite=True)