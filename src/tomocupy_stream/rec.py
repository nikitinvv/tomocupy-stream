#!/usr/bin/env python
# -*- coding: utf-8 -*-

# *************************************************************************** #
#                  Copyright © 2022, UChicago Argonne, LLC                    #
#                           All Rights Reserved                               #
#                         Software Name: Tomocupy_stream                             #
#                     By: Argonne National Laboratory                         #
#                                                                             #
#                           OPEN SOURCE LICENSE                               #
#                                                                             #
# Redistribution and use in source and binary forms, with or without          #
# modification, are permitted provided that the following conditions are met: #
#                                                                             #
# 1. Redistributions of source code must retain the above copyright notice,   #
#    this list of conditions and the following disclaimer.                    #
# 2. Redistributions in binary form must reproduce the above copyright        #
#    notice, this list of conditions and the following disclaimer in the      #
#    documentation and/or other materials provided with the distribution.     #
# 3. Neither the name of the copyright holder nor the names of its            #
#    contributors may be used to endorse or promote products derived          #
#    from this software without specific prior written permission.            #
#                                                                             #
#                                                                             #
# *************************************************************************** #
#                               DISCLAIMER                                    #
#                                                                             #
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS         #
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT           #
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS           #
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT    #
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,      #
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED    #
# TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR      #
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF      #
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING        #
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS          #
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.                #
# *************************************************************************** #

from tomocupy_stream import utils
from tomocupy_stream import tomo_functions
import cupy as cp
import numpy as np
import threading

__author__ = "Viktor Nikitin"
__copyright__ = "Copyright (c) 2023, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['GPURecRAM', ]

class GPURecRAM:
    '''
    Class for tomographic reconstruction on GPU with conveyor data processing by sinogram chunks (in z direction).
    CUDA Streams are used to overlap CPU-GPU data transfers with computations.    
    '''

    @classmethod
    def for_data_like(
        cls,
        *,
        data,
        dark,
        flat,
        ncz,
        dtype,
        rotation_axis,
        reconstruction_algorithm="fourierrec",
        remove_stripe_method="None",
        fw_sigma=None,
        fw_filter=None,
        fw_level=None,
        ti_beta=None,
        ti_mask=None,
        dezinger=0,
        dezinger_threshold=5000,
        fbp_filter="parzen",
    ):
        """
        Construct a GPURecRAM instance from sample data.
        """
        return cls(
            n=data.shape[2],
            nz=data.shape[1],
            nproj=data.shape[0],
            ncz=ncz,
            ndark=dark.shape[0],
            nflat=flat.shape[0],
            in_dtype=data.dtype,
            dtype=dtype,
            rotation_axis=rotation_axis,
            reconstruction_algorithm=reconstruction_algorithm,
            remove_stripe_method=remove_stripe_method,
            fw_sigma=fw_sigma,
            fw_filter=fw_filter,
            fw_level=fw_level,
            ti_beta=ti_beta,
            ti_mask=ti_mask,
            dezinger=dezinger,
            dezinger_threshold=dezinger_threshold,
            fbp_filter=fbp_filter,
        )


    def __init__(
        self,
        *,
        n,
        nz,
        nproj,
        ncz,
        ndark,
        nflat,
        in_dtype,
        dtype,
        rotation_axis,
        reconstruction_algorithm="fourierrec",
        remove_stripe_method="None",
        fw_sigma=None,
        fw_filter=None,
        fw_level=None,
        ti_beta=None,
        ti_mask=None,
        dezinger=0,
        dezinger_threshold=5000,
        fbp_filter="parzen",
    ):
        """
        n : int
            Smaple size in x
        nz : int
            Sample size in z
        nproj : int
            Number of projection angles
        ncz : int
            Chunk size (must be multiple of 2)
        ndark : int
            Number of dark fields
        nflat : int
            Number of flat fields
        in_dtype : numpy.dtype or str
            Input data type
        dtype : numpy.dtype or str
            Output data type
        rotation_axis : float
        reconstruction_algorithm : {"fourierrec", "lprec", "linerec"}
        remove_stripe_method : {"fw", "ti"}
        fw_sigma : float
            Only applicable if remove_stripe_method="fw"
        fw_filter :
            Only applicable if remove_stripe_method="fw"
        fw_level : float
            Only applicable if remove_stripe_method="fw"
        ti_beta : float
            Only applicable if remove_stripe_method="ti"
        ti_mask : float
            Only applicable if remove_stripe_method="ti"
        dezinger : int
            0 means no zingers; 2 means remove zingers
        dezinger_threshold : float
        fbp_filter :
            Default is "parzen"

        """
        if (ncz % 2 != 0):
            raise ValueError("Chunk size must be a multiple of 2")
        
        self.n = n
        self.nz = nz
        self.nproj = nproj
        self.ncz = ncz
        self.ndark = ndark
        self.nflat = nflat
        self.dtype = dtype
        self.shape_data_chunk = (ncz, nproj, n)
        self.shape_recon_chunk = (ncz, n, n)
        self.shape_dark_chunk = (ndark, ncz, n)
        self.shape_flat_chunk = (nflat, ncz, n)
        self.tomofunc_kwargs = dict(
            n=n,
            nproj=nproj,
            ncz=ncz,
            dtype=dtype,
            rotation_axis=rotation_axis,
            reconstruction_algorithm=reconstruction_algorithm,
            remove_stripe_method=remove_stripe_method,
            fw_sigma=fw_sigma,
            fw_filter=fw_filter,
            fw_level=fw_level,
            ti_beta=ti_beta,
            ti_mask=ti_mask,
            dezinger=dezinger,
            dezinger_threshold=dezinger_threshold,
            fbp_filter=fbp_filter,
        )
        
        # pinned memory for data item
        self.item_pinned = {}
        self.item_pinned['data'] = utils.pinned_array(np.zeros([2, *self.shape_data_chunk], dtype=in_dtype))
        self.item_pinned['dark'] = utils.pinned_array(np.zeros([2, *self.shape_dark_chunk], dtype=in_dtype))
        self.item_pinned['flat'] = utils.pinned_array(np.ones([2, *self.shape_flat_chunk], dtype=in_dtype))

        # gpu memory for data item
        self.item_gpu = {}
        self.item_gpu['data'] = cp.zeros([2, *self.shape_data_chunk], dtype=in_dtype)
        self.item_gpu['dark'] = cp.zeros([2, *self.shape_dark_chunk], dtype=in_dtype)
        self.item_gpu['flat'] = cp.ones([2, *self.shape_flat_chunk], dtype=in_dtype)

        # pinned memory for reconstrution
        self.rec_pinned = utils.pinned_array(np.zeros([16,*self.shape_recon_chunk], dtype=dtype))

        # gpu memory for reconstrution
        self.rec_gpu = cp.zeros([2, *self.shape_recon_chunk], dtype=dtype)
                
        # chunks        
        self.nzchunk = int(np.ceil(nz/ncz))
        self.lzchunk = np.minimum(
            ncz, np.int32(nz-np.arange(self.nzchunk)*ncz))  # chunk sizes        
        self.ncz = ncz
        self.nz = nz
        
        # streams for overlapping data transfers with computations
        self.stream1 = cp.cuda.Stream(non_blocking=False)
        self.stream2 = cp.cuda.Stream(non_blocking=False)
        self.stream3 = cp.cuda.Stream(non_blocking=False)
        
        # threads for filling the resulting array
        self.write_threads = []
        for k in range(16):#16 is probably enough but can be changed
            self.write_threads.append(utils.WRThread())
        
    def recon_all(self, data, dark, flat, theta, output=None):
        # Validate that the inputs match what was declared in __init__ and pre-allocated for.
        expected_data_shape = (self.nz, self.nproj, self.n)
        if data.shape != expected_data_shape:
            raise ValueError(f"Expected data.shape {expected_data_shape}, got {data.shape}")
        expected_dark_shape = (self.ndark, self.nz, self.n)
        if dark.shape != expected_dark_shape:
            raise ValueError(f"Expected dark.shape {expected_dark_shape}, got {dark.shape}")
        expected_flat_shape = (self.nflat, self.nz, self.n)
        if flat.shape != expected_flat_shape:
            raise ValueError(f"Expected flat.shape {expected_flat_shape}, got {flat.shape}")
        if output is None:
            # Allocate output array.
            output = np.zeros([self.nz, self.n, self.n], dtype=self.dtype)
        else:
            # Use pre-allocated output array, first validating it.
            expected_output_shape = (self.nz, self.n, self.n)
            if output.shape != expected_output_shape:
                raise ValueError(f"Expected output.shape {expected_output_shape}, got {output.shape}")
                
        self.cl_tomo_func = tomo_functions.TomoFunctions(theta=theta, **self.tomofunc_kwargs)
        
        # Pipeline for data cpu-gpu copy and reconstruction
        for k in range(self.nzchunk+2):
            if k<self.nzchunk:
                utils.printProgressBar(k, self.nzchunk, length=40)
            if(k > 0 and k < self.nzchunk+1):
                with self.stream2:  # reconstruction
                    data_chunk = self.item_gpu['data'][(k-1) % 2]
                    dark_chunk = self.item_gpu['dark'][(k-1) % 2]
                    flat_chunk = self.item_gpu['flat'][(k-1) % 2]
                    
                    rec = self.rec_gpu[(k-1) % 2]                    
                    self.cl_tomo_func.rec(rec, data_chunk, dark_chunk, flat_chunk)
                    
            if(k > 1):
                with self.stream3:  # gpu->cpu copy
                    # find free thread
                    ithread = utils.find_free_thread(self.write_threads)
                    self.rec_gpu[(k-2) % 2].get(out=self.rec_pinned[ithread])                    
                    
            if(k < self.nzchunk):
                # copy to pinned memory
                self.item_pinned['data'][k % 2, :self.lzchunk[k]] = data[k*self.ncz:k*self.ncz+self.lzchunk[k]]
                self.item_pinned['dark'][k % 2, :, :self.lzchunk[k]] = dark[:,k*self.ncz:k*self.ncz+self.lzchunk[k]]
                self.item_pinned['flat'][k % 2, :, :self.lzchunk[k]] = flat[:,k*self.ncz:k*self.ncz+self.lzchunk[k]]
                
                
                with self.stream1:  # cpu->gpu copy
                    self.item_gpu['data'][k % 2].set(self.item_pinned['data'][k % 2])
                    self.item_gpu['dark'][k % 2].set(self.item_pinned['dark'][k % 2])
                    self.item_gpu['flat'][k % 2].set(self.item_pinned['flat'][k % 2])
            self.stream3.synchronize()
            if(k > 1): 
                # run a thread filling the resulting array (after gpu->cpu copy is done)
                st = (k-2)*self.ncz
                end = st+self.lzchunk[k-2]
                self.write_threads[ithread].run(utils.fill_array, (output, self.rec_pinned[ithread], st, end))

            self.stream1.synchronize()
            self.stream2.synchronize()
            
        for t in self.write_threads:
            t.join()
        return output 
