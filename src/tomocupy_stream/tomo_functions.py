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

from tomocupy_stream import fourierrec
from tomocupy_stream import lprec
from tomocupy_stream import fbp_filter as fbp_filter_module
from tomocupy_stream import linerec
from tomocupy_stream import remove_stripe
import cupyx.scipy.ndimage as ndimage


import cupy as cp
import numpy as np


class TomoFunctions():
    def __init__(
        self,
        *,
        n,
        nproj,
        ncz,
        dtype,
        rotation_axis,
        reconstruction_algorithm,
        remove_stripe_method,
        fw_sigma,
        fw_filter,
        fw_level,
        ti_beta,
        ti_mask,
        dezinger,
        dezinger_threshold,
        fbp_filter
    ):
        self.n = n
        self.nproj = nproj
        self.ncz = ncz
        self.dtype = dtype
        self.rotation_axis = rotation_axis
        self.reconstruction_algorithm = reconstruction_algorithm
        self.remove_stripe_method = remove_stripe_method
        self.fw_sigma = fw_sigma
        self.fw_filter = fw_filter
        self.fw_level = fw_level
        self.ti_beta = ti_beta
        self.ti_mask = ti_mask
        self.dezinger = dezinger
        self.dezinger_threshold = dezinger_threshold
        self.fbp_filter = fbp_filter

        # padded size for filtering
        self.ne = 3*self.n//2
        if self.dtype == 'float16':
            # power of 2 for float16
            self.ne = 2**int(np.ceil(np.log2(3*self.n//2)))

        # filter class
        self.cl_filter = fbp_filter_module.FBPFilter(
            self.ne, self.nproj, self.ncz, self.dtype)

        # backprojection class
        if self.reconstruction_algorithm == 'fourierrec':
            self.cl_rec = fourierrec.FourierRec(
                self.n, self.nproj, self.ncz, self.dtype)
        elif self.reconstruction_algorithm == 'lprec':
            self.rotation_axis += 0.5
            self.cl_rec = lprec.LpRec(
                self.n, self.nproj, self.ncz, self.dtype)
        elif self.reconstruction_algorithm == 'linerec':
            self.cl_rec = linerec.LineRec(
                self.nproj, self.nproj, self.ncz, self.ncz, self.n, self.dtype)

    def rec(self, result, data, dark, flat, theta):
        """Processing a sinogram data chunk"""

        self._remove_outliers(data)
        self._remove_outliers(dark)
        self._remove_outliers(flat)
        tmp = self._darkflat_correction(data, dark, flat)  # new memory -> tmp
        if self.remove_stripe_method == 'fw':
            remove_stripe.remove_stripe_fw(
                tmp, self.fw_sigma, self.fw_filter, self.fw_level)
        elif self.remove_stripe_method == 'ti':
            remove_stripe.remove_stripe_ti(
                tmp, self.ti_beta, self.ti_mask)

        self._minus_log(tmp)
        self._fbp_filter_center(tmp)
        self.cl_rec.backprojection(
            result, tmp, theta, cp.cuda.get_current_stream())

    def _darkflat_correction(self, data, dark, flat):
        """Dark-flat field correction"""

        dark0 = dark.astype(self.dtype, copy=False)
        flat0 = flat.astype(self.dtype, copy=False)
        flat0 = cp.mean(flat0, axis=0)[:, np.newaxis]
        dark0 = cp.mean(dark0, axis=0)[:, np.newaxis]
        res = (data.astype(self.dtype, copy=False)-dark0) / (flat0-dark0+1e-3)
        res[res <= 0] = 1
        return res

    def _minus_log(self, data):
        """Taking negative logarithm"""

        data[:] = -cp.log(data)
        data[cp.isnan(data)] = 6.0
        data[cp.isinf(data)] = 0

    def _remove_outliers(self, data):
        """Remove outliers"""

        if (int(self.dezinger) > 0):
            w = int(self.dezinger)
            if len(data.shape) == 3:
                fdata = ndimage.median_filter(data, [w, 1, w])
            else:
                fdata = ndimage.median_filter(data, [w, w])
            data[:] = cp.where(cp.logical_and(
                data > fdata, (data - fdata) > self.dezinger_threshold), fdata, data)

    def _fbp_filter_center(self, data):
        """FBP filtering of projections with applying the rotation center shift wrt to the origin"""

        t = cp.fft.rfftfreq(self.ne).astype('float32')
        if self.fbp_filter == 'parzen':
            w = t * (1 - t * 2)**3
        elif self.fbp_filter == 'shepp':
            w = t * cp.sinc(t)

        tmp = cp.pad(
            data, ((0, 0), (0, 0), (self.ne//2-self.n//2, self.ne//2-self.n//2)), mode='edge')
        # center fix
        w = w*cp.exp(-2*cp.pi*1j*t*(-self.rotation_axis + self.n/2))
        self.cl_filter.filter(tmp, w, cp.cuda.get_current_stream())
        data[:] = tmp[:, :, self.ne//2-self.n//2:self.ne//2+self.n//2]
