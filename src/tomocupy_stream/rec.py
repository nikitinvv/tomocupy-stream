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

class GPURecRAM():
    '''
    Class for tomographic reconstruction on GPU with conveyor data processing by sinogram chunks (in z direction).
    CUDA Streams are used to overlap CPU-GPU data transfers with computations.    
    '''

    def __init__(self, args, theta):                
        
        self.shape_data_chunk = (args.ncz, args.nproj, args.n)
        self.shape_recon_chunk = (args.ncz, args.n, args.n)
        self.shape_dark_chunk = (args.ndark, args.ncz, args.n)
        self.shape_flat_chunk = (args.nflat, args.ncz, args.n)
        
        # pinned memory for data item
        self.item_pinned = {}
        self.item_pinned['data'] = utils.pinned_array(np.zeros([2, *self.shape_data_chunk], dtype=args.in_dtype))
        self.item_pinned['dark'] = utils.pinned_array(np.zeros([2, *self.shape_dark_chunk], dtype=args.in_dtype))
        self.item_pinned['flat'] = utils.pinned_array(np.ones([2, *self.shape_flat_chunk], dtype=args.in_dtype))

        # gpu memory for data item
        self.item_gpu = {}
        self.item_gpu['data'] = cp.zeros([2, *self.shape_data_chunk], dtype=args.in_dtype)
        self.item_gpu['dark'] = cp.zeros([2, *self.shape_dark_chunk], dtype=args.in_dtype)
        self.item_gpu['flat'] = cp.ones([2, *self.shape_flat_chunk], dtype=args.in_dtype)

        # pinned memory for reconstrution
        self.rec_pinned = utils.pinned_array(np.zeros([16,*self.shape_recon_chunk], dtype=args.dtype))

        # gpu memory for reconstrution
        self.rec_gpu = cp.zeros([2, *self.shape_recon_chunk], dtype=args.dtype)
                
        # chunks        
        self.nzchunk = int(np.ceil(args.nz/args.ncz))
        self.lzchunk = np.minimum(
            args.ncz, np.int32(args.nz-np.arange(self.nzchunk)*args.ncz))  # chunk sizes        
        self.ncz = args.ncz
        self.nz = args.nz
        
        # streams for overlapping data transfers with computations
        self.stream1 = cp.cuda.Stream(non_blocking=False)
        self.stream2 = cp.cuda.Stream(non_blocking=False)
        self.stream3 = cp.cuda.Stream(non_blocking=False)
        
        self.cl_tomo_func = tomo_functions.TomoFunctions(args,theta)
        
        # threads for filling the resulting array
        self.write_threads = []
        for k in range(16):#16 is probably enough but can be changed
            self.write_threads.append(utils.WRThread())
        
    def recon_all(self, result, data_in, dark_in, flat_in):
                
        # Pipeline for data cpu-gpu copy and reconstruction
        for k in range(self.nzchunk+2):
            if k<self.nzchunk:
                utils.printProgressBar(k, self.nzchunk, length=40)
            if(k > 0 and k < self.nzchunk+1):
                with self.stream2:  # reconstruction
                    data = self.item_gpu['data'][(k-1) % 2]
                    dark = self.item_gpu['dark'][(k-1) % 2]
                    flat = self.item_gpu['flat'][(k-1) % 2]
                    
                    rec = self.rec_gpu[(k-1) % 2]                    
                    self.cl_tomo_func.rec(rec, data, dark, flat)
                    
            if(k > 1):
                with self.stream3:  # gpu->cpu copy
                    # find free thread
                    ithread = utils.find_free_thread(self.write_threads)
                    self.rec_gpu[(k-2) % 2].get(out=self.rec_pinned[ithread])                    
                    
            if(k < self.nzchunk):
                # copy to pinned memory
                self.item_pinned['data'][k % 2, :self.lzchunk[k]] = data_in[k*self.ncz:k*self.ncz+self.lzchunk[k]]
                self.item_pinned['dark'][k % 2, :, :self.lzchunk[k]] = dark_in[:,k*self.ncz:k*self.ncz+self.lzchunk[k]]
                self.item_pinned['flat'][k % 2, :, :self.lzchunk[k]] = flat_in[:,k*self.ncz:k*self.ncz+self.lzchunk[k]]
                
                
                with self.stream1:  # cpu->gpu copy
                    self.item_gpu['data'][k % 2].set(self.item_pinned['data'][k % 2])
                    self.item_gpu['dark'][k % 2].set(self.item_pinned['dark'][k % 2])
                    self.item_gpu['flat'][k % 2].set(self.item_pinned['flat'][k % 2])
            self.stream3.synchronize()
            if(k > 1): 
                # run a thread filling the resulting array (after gpu->cpu copy is done)
                st = (k-2)*self.ncz
                end = st+self.lzchunk[k-2]
                self.write_threads[ithread].run(utils.fill_array, (result, self.rec_pinned[ithread], st, end))

            self.stream1.synchronize()
            self.stream2.synchronize()
            
        for t in self.write_threads:
            t.join()
            