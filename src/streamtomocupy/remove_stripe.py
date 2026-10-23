import cupy as cp
import pywt
from functools import lru_cache
from cupyx.scipy.ndimage import median_filter
from cupyx.scipy.ndimage import binary_dilation
from cupyx.scipy.ndimage import uniform_filter1d

__all__ = [
    "DWTForward",
    "DWTInverse",
    "afb1d",
    "remove_all_stripe",
    "remove_stripe_fw",
    "remove_stripe_ti",
]

###### Ring removal with wavelet filtering (adapted for cupy from pytroch_wavelet package https://pytorch-wavelets.readthedocs.io/)##########


# One thread per output sample, tap loop in registers: the array is read once
# instead of once per tap, and the symmetric padding is folded in.

# pywt 'symmetric' reflection of t into [0, N)
_REFLECT = r"""
    if (t < 0 || t >= N) {
        int p = 2 * N;
        int m = t % p;
        if (m < 0) m += p;
        t = (m < N) ? m : (p - 1 - m);
    }
"""

_afb_row = cp.ElementwiseKernel(
    "raw float32 x, raw float32 h0, raw float32 h1, "
    "int32 L, int32 N, int32 OS, int32 M1, int32 H",
    "raw float32 out",
    """
    int o   = i % OS;
    int bch = i / OS;                   // (b*C + c)*H + h
    int h   = bch % H;
    long long base = (long long)bch * N;

    int t0 = 2 * o - M1;
    float a = 0.f, b = 0.f;
    if (t0 >= 0 && t0 + L <= N) {       // interior: no reflection needed
        for (int j = 0; j < L; ++j) {
            float v = x[base + t0 + j];
            a += h0[j] * v;  b += h1[j] * v;
        }
    } else {
        for (int j = 0; j < L; ++j) {
            int t = t0 + j;
"""
    + _REFLECT
    + """
            float v = x[base + t];
            a += h0[j] * v;  b += h1[j] * v;
        }
    }
    long long ob = ((long long)(bch / H) * 2 * H + h) * OS + o;
    out[ob]                     = a;    // lowpass
    out[ob + (long long)H * OS] = b;    // highpass
    """,
    "afb_row",
)

_afb_col = cp.ElementwiseKernel(
    "raw float32 x, raw float32 h0, raw float32 h1, "
    "int32 L, int32 N, int32 OS, int32 M1, int32 W",
    "raw float32 out",
    """
    int w  = i % W;                     // fastest axis, so loads coalesce
    int r  = i / W;
    int o  = r % OS;
    long long bc = r / OS;              // b*C + c
    long long base = bc * (long long)N * W + w;

    int t0 = 2 * o - M1;
    float a = 0.f, b = 0.f;
    if (t0 >= 0 && t0 + L <= N) {
        for (int j = 0; j < L; ++j) {
            float v = x[base + (long long)(t0 + j) * W];
            a += h0[j] * v;  b += h1[j] * v;
        }
    } else {
        for (int j = 0; j < L; ++j) {
            int t = t0 + j;
"""
    + _REFLECT
    + """
            float v = x[base + (long long)t * W];
            a += h0[j] * v;  b += h1[j] * v;
        }
    }
    long long ob = (bc * 2 * OS + o) * W + w;
    out[ob]                     = a;
    out[ob + (long long)OS * W] = b;
    """,
    "afb_col",
)


def afb1d(x, h0, h1="zero", dim=-1):
    """1D analysis filter bank: stride-2 convolution along dim, all channels in parallel.

    Output channels interleaved [lo_0, hi_0, lo_1, hi_1, ...] to match
    the grouped-convolution ordering expected by DWTForward.
    """
    C = x.shape[1]
    d = dim % 4
    N = x.shape[d]
    h0f = cp.ascontiguousarray(h0.ravel(), dtype="float32")
    h1f = cp.ascontiguousarray(h1.ravel(), dtype="float32")
    L = int(h0f.size)
    outsize = pywt.dwt_coeff_len(N, L, mode="symmetric")
    # left pad width; the right pad only has to exist, the kernel never reads
    # past the taps it needs
    m1 = (2 * (outsize - 1) - N + L) // 2
    x = cp.ascontiguousarray(x, dtype="float32")
    B = x.shape[0]
    if d == 3:  # row direction: stride-2 along axis 3
        H = x.shape[2]
        out = cp.empty((B, C, 2, H, outsize), dtype="float32")
        _afb_row(x, h0f, h1f, L, N, outsize, m1, H, out, size=B * C * H * outsize)
    else:  # col direction: stride-2 along axis 2
        W = x.shape[3]
        out = cp.empty((B, C, 2, outsize, W), dtype="float32")
        _afb_col(x, h0f, h1f, L, N, outsize, m1, W, out, size=B * C * outsize * W)
    return out.reshape(B, 2 * C, *out.shape[3:])


# Upsample-and-filter inverted from a scatter into a gather, so each output is
# written once: out[q] = sum_{j = q (mod 2)} g0[j]*lo[(q-j)/2] + g1[j]*hi[(q-j)/2]
_sfb_row = cp.ElementwiseKernel(
    "raw float32 lo, raw float32 hi, raw float32 g0, raw float32 g1, "
    "int32 L, int32 W, int32 OW",
    "float32 out",
    """
    int q = i % OW + L - 2;
    long long base = (long long)(i / OW) * W;
    float a = 0.f;
    for (int j = q & 1; j < L; j += 2) {
        int w = (q - j) >> 1;
        if (w >= 0 && w < W) a += g0[j] * lo[base + w] + g1[j] * hi[base + w];
    }
    out = a;
    """,
    "sfb_row",
)

_sfb_col = cp.ElementwiseKernel(
    "raw float32 lo, raw float32 hi, raw float32 g0, raw float32 g1, "
    "int32 L, int32 H, int32 OH, int32 W",
    "float32 out",
    """
    int w = i % W;
    int r = i / W;
    int q = r % OH + L - 2;
    long long base = (long long)(r / OH) * H * W + w;
    float a = 0.f;
    for (int j = q & 1; j < L; j += 2) {
        int h = (q - j) >> 1;
        if (h >= 0 && h < H)
            a += g0[j] * lo[base + (long long)h * W]
               + g1[j] * hi[base + (long long)h * W];
    }
    out = a;
    """,
    "sfb_col",
)


def sfb1d(lo, hi, g0, g1="zero", dim=-1):
    """1D synthesis filter bank: upsample by 2 along dim and filter."""
    C = lo.shape[1]
    d = dim % 4
    g0f = cp.ascontiguousarray(g0.ravel(), dtype="float32")
    g1f = cp.ascontiguousarray(g1.ravel(), dtype="float32")
    L = int(g0f.size)
    B = lo.shape[0]
    lo = cp.ascontiguousarray(lo, dtype="float32")
    hi = cp.ascontiguousarray(hi, dtype="float32")
    H, W = lo.shape[2], lo.shape[3]
    if d == 3:  # row direction: stride (1, 2)
        ow = (W - 1) * 2 + L - 2 * (L - 2)
        out = cp.empty((B, C, H, ow), dtype="float32")
        _sfb_row(lo, hi, g0f, g1f, L, W, ow, out)
    else:  # col direction: stride (2, 1)
        oh = (H - 1) * 2 + L - 2 * (L - 2)
        out = cp.empty((B, C, oh, W), dtype="float32")
        _sfb_col(lo, hi, g0f, g1f, L, H, oh, W, out)
    return out


class DWTForward:
    """Performs a 2d DWT Forward decomposition of an image

    Args:
        wave (str): Which wavelet to use.
    """

    def __init__(self, wave="db1"):
        super().__init__()

        wave = pywt.Wavelet(wave)
        h0_col, h1_col = wave.dec_lo, wave.dec_hi
        h0_row, h1_row = h0_col, h1_col

        self.h0_col = cp.array(h0_col).astype("float32")[::-1].reshape((1, 1, -1, 1))
        self.h1_col = cp.array(h1_col).astype("float32")[::-1].reshape((1, 1, -1, 1))
        self.h0_row = cp.array(h0_row).astype("float32")[::-1].reshape((1, 1, 1, -1))
        self.h1_row = cp.array(h1_row).astype("float32")[::-1].reshape((1, 1, 1, -1))

    def apply(self, x):
        """Forward pass of the DWT.

        Args:
            x (array): Input of shape :math:`(N, C_{in}, H_{in}, W_{in})`

        Returns:
            (yl, yh)
                tuple of lowpass (yl) and bandpass (yh) coefficients.
                yh is a list of scale coefficients. yl has shape
                :math:`(N, C_{in}, H_{in}', W_{in}')` and yh has shape
                :math:`list(N, C_{in}, 3, H_{in}'', W_{in}'')`. The new
                dimension in yh iterates over the LH, HL and HH coefficients.

        Note:
            :math:`H_{in}', W_{in}', H_{in}'', W_{in}''` denote the correctly
            downsampled shapes of the DWT pyramid.
        """
        # Do a multilevel transform
        # Do 1 level of the transform
        lohi = afb1d(x, self.h0_row, self.h1_row, dim=3)
        y = afb1d(lohi, self.h0_col, self.h1_col, dim=2)
        s = y.shape
        y = y.reshape(s[0], -1, 4, s[-2], s[-1])
        x = cp.ascontiguousarray(y[:, :, 0])
        yh = cp.ascontiguousarray(y[:, :, 1:])
        return x, yh


class DWTInverse:
    """Performs a 2d DWT Inverse reconstruction of an image

    Args:
        wave (str): Which wavelet to use.
    """

    def __init__(self, wave="db1"):
        super().__init__()
        wave = pywt.Wavelet(wave)
        g0_col, g1_col = wave.rec_lo, wave.rec_hi
        g0_row, g1_row = g0_col, g1_col
        # Prepare the filters
        self.g0_col = cp.array(g0_col).astype("float32").reshape((1, 1, -1, 1))
        self.g1_col = cp.array(g1_col).astype("float32").reshape((1, 1, -1, 1))
        self.g0_row = cp.array(g0_row).astype("float32").reshape((1, 1, 1, -1))
        self.g1_row = cp.array(g1_row).astype("float32").reshape((1, 1, 1, -1))

    def apply(self, coeffs):
        """
        Args:
            coeffs (yl, yh): tuple of lowpass and bandpass coefficients, where:
              yl is a lowpass array of shape :math:`(N, C_{in}, H_{in}',
              W_{in}')` and yh is a list of bandpass arrays of shape
              :math:`list(N, C_{in}, 3, H_{in}'', W_{in}'')`. I.e. should match
              the format returned by DWTForward

        Returns:
            Reconstructed input of shape :math:`(N, C_{in}, H_{in}, W_{in})`

        Note:
            :math:`H_{in}', W_{in}', H_{in}'', W_{in}''` denote the correctly
            downsampled shapes of the DWT pyramid.

        """
        yl, yh = coeffs
        # the two independent dim=2 calls batched into one C=2 call
        lo_hi = sfb1d(
            cp.concatenate([yl, yh[:, :, 1]], axis=1),
            cp.concatenate([yh[:, :, 0], yh[:, :, 2]], axis=1),
            self.g0_col,
            self.g1_col,
            dim=2,
        )  # [B, 2, H, W']
        yl = sfb1d(lo_hi[:, :1], lo_hi[:, 1:], self.g0_row, self.g1_row, dim=3)
        return yl


# the caches hand back device arrays, so the device is part of the key


@lru_cache(maxsize=64)
def _fw_damp_dev(my, sigma, device):
    y_hat = cp.fft.ifftshift((cp.arange(-my, my, 2) + 1) / 2)
    damp = -cp.expm1(-(y_hat**2) / (2 * sigma**2))
    # y_hat is half-sample offset, so damp[-k] != damp[k]; the full-fft form
    # took .real, which used only its even part. Symmetrize to match.
    k = cp.arange(my // 2 + 1)
    return 0.5 * (damp[k] + damp[(-k) % my])


def _fw_damp(my, sigma):
    """Damping profile for a band of `my` angles, in rfft layout."""
    return _fw_damp_dev(my, sigma, cp.cuda.runtime.getDevice())


@lru_cache(maxsize=16)
def _fw_transforms_dev(wname, device):
    return DWTForward(wave=wname), DWTInverse(wave=wname)


def _fw_transforms(wname):
    """Filter banks for a wavelet; they hold only taps, so share them."""
    return _fw_transforms_dev(wname, cp.cuda.runtime.getDevice())


def remove_stripe_fw(data, sigma, wname, level):
    """Remove stripes with wavelet filtering"""

    [nproj, nz, ni] = data.shape

    nproj_pad = nproj + nproj // 8

    # Accepts all wave types available to PyWavelets
    xfm, ifm = _fw_transforms(wname)

    # Wavelet decomposition.
    cc = []
    sli = cp.zeros([nz, 1, nproj_pad, ni], dtype="float32")

    sli[:, 0, (nproj_pad - nproj) // 2 : (nproj_pad + nproj) // 2] = data.astype(
        "float32"
    ).swapaxes(0, 1)
    for k in range(level):
        sli, c = xfm.apply(sli)
        cc.append(c)
        # FFT - use rfft (real input -> ~2x faster, half memory)
        band = cc[k][:, 0, 1]
        _, my, mx = band.shape
        fcV = cp.fft.rfft(band, axis=1)  # [nz, my//2+1, mx]
        # Damping of ring artifact information.
        fcV *= _fw_damp(my, float(sigma))[:, None]
        # Inverse FFT.
        cc[k][:, 0, 1] = cp.fft.irfft(fcV, my, axis=1)

    # # Wavelet reconstruction.
    for k in range(level)[::-1]:
        shape0 = cc[k][0, 0, 1].shape
        sli = sli[:, :, : shape0[0], : shape0[1]]
        sli = ifm.apply((sli, cc[k]))

    data = sli[:, 0, (nproj_pad - nproj) // 2 : (nproj_pad + nproj) // 2, :ni].astype(
        data.dtype
    )  # modified
    data = data.swapaxes(0, 1)

    return data


######## Titarenko ring removal ################
def remove_stripe_ti(data, beta, mask_size):
    """Remove stripes with a new method by V. Titareno"""
    gamma = beta * ((1 - beta) / (1 + beta)) ** cp.abs(
        cp.fft.fftfreq(data.shape[-1]) * data.shape[-1]
    )
    gamma[0] -= 1
    v = cp.mean(data, axis=0)
    v = v - v[:, 0:1]
    v = cp.fft.irfft(cp.fft.rfft(v) * cp.fft.rfft(gamma))
    mask = cp.zeros(v.shape, dtype=v.dtype)
    mask_size = mask_size * mask.shape[1]
    mask[
        :, mask.shape[1] // 2 - mask_size // 2 : mask.shape[1] // 2 + mask_size // 2
    ] = 1
    data[:] += v * mask
    return data


######## Optimized version for Vo-all ring removal in tomopy#########
def _detect_stripe_batch(listdata, snr):
    """Locate stripes (algorithm 4 in :cite:`Vo:18`), batched over [nz, ni]."""
    nz, numdata = listdata.shape
    listsorted = cp.sort(listdata, axis=1)[:, ::-1]
    xlist = cp.arange(numdata, dtype="float32")
    ndrop = int(0.25 * numdata)
    x = xlist[ndrop : -ndrop - 1]
    y = listsorted[:, ndrop : -ndrop - 1]
    n = x.size
    xm = float(x.mean())
    ym = y.mean(axis=1)  # [nz]
    Sxy = (y * x).sum(axis=1) - n * ym * xm  # [nz]
    Sxx = float((x**2).sum() - n * xm**2)
    slope = Sxy / Sxx  # [nz]
    intercept = ym - slope * xm  # [nz]
    numt1 = intercept + slope * xlist[-1]  # [nz]
    noiselevel = cp.clip(cp.abs(numt1 - intercept), 1e-6, None)  # [nz]
    val1 = cp.abs(listsorted[:, 0] - intercept) / noiselevel  # [nz]
    val2 = cp.abs(listsorted[:, -1] - numt1) / noiselevel  # [nz]
    listmask = cp.zeros((nz, numdata), dtype="float32")
    upper = (intercept + noiselevel * snr * 0.5)[:, None]  # [nz, 1]
    lower = (numt1 - noiselevel * snr * 0.5)[:, None]  # [nz, 1]
    listmask = cp.where((val1 >= snr)[:, None] & (listdata > upper), 1.0, listmask)
    listmask = cp.where((val2 >= snr)[:, None] & (listdata <= lower), 1.0, listmask)
    return listmask


# float32 bits as uint32 with the position in the low half, so one uint64
# radix sort replaces argsort. -0.0 and NaNs are remapped first.
_pack_key = cp.ElementwiseKernel(
    "float32 x, int32 n",
    "uint64 out",
    """
    unsigned int u = __float_as_uint(x);
    unsigned int k;
    if (x != x) k = 0xffffffffu;                    // every NaN sorts last
    else if (u == 0x80000000u) k = 0x80000000u;     // -0.0 reads as +0.0
    else k = (u >> 31) ? ~u : (u | 0x80000000u);
    out = ((unsigned long long)k << 32) | (unsigned long long)(i % n);
    """,
    "_rs_pack_key",
)

_unpack_key = cp.ElementwiseKernel(
    "uint64 p",
    "float32 val, int64 idx",
    """
    unsigned int k = (unsigned int)(p >> 32);
    unsigned int u = (k >> 31) ? (k & 0x7fffffffu) : ~k;
    val = __uint_as_float(u);
    idx = (long long)(p & 0xffffffffu);
    """,
    "_rs_unpack_key",
)


def _sort_with_ids3(x):
    """Sorted values along axis 2 and the positions they came from.

    Ties keep their original order, so this matches a stable argsort.
    """
    x = cp.ascontiguousarray(x, dtype="float32")
    packed = _pack_key(x, cp.int32(x.shape[2]))
    packed.sort(axis=2)
    return _unpack_key(packed)


def _unsort3(ids, vals):
    """Scatter vals back to the positions ids came from, along axis=2."""
    if hasattr(cp, "put_along_axis"):
        out = cp.empty_like(vals)
        cp.put_along_axis(out, ids, vals, axis=2)
        return out
    # cupy<13: argsort of a permutation is its inverse permutation
    return cp.take_along_axis(vals, cp.argsort(ids, axis=2), axis=2)


def _rs_sort3(tomo, size, dim):
    """Batched _rs_sort for 3D input [nproj, nz, ni]."""
    t = cp.transpose(tomo, (2, 1, 0))  # [ni, nz, nproj]
    matsort_vals, ids = _sort_with_ids3(t)
    del t
    if dim == 1:
        matsort_vals = median_filter(matsort_vals, (size, 1, 1))
    else:
        matsort_vals = median_filter(matsort_vals, (size, 1, size))
    return cp.transpose(_unsort3(ids, matsort_vals), (2, 1, 0))


def _rs_large3(tomo, snr, size, drop_ratio=0.1, norm=True):
    """Batched _rs_large for 3D input [nproj, nz, ni]."""
    drop_ratio = max(min(drop_ratio, 0.8), 0)
    nproj, nz, ni = tomo.shape
    ndrop = int(0.5 * drop_ratio * nproj)
    # normalization divides each nproj-column by a scalar, preserving sort
    # order, so ids stays valid after it
    t = cp.transpose(tomo, (2, 1, 0))  # [ni, nz, nproj]
    sortvals, ids = _sort_with_ids3(t)
    sinosort = cp.transpose(sortvals, (2, 1, 0))  # [nproj, nz, ni]
    del t, sortvals
    sinosmooth = median_filter(sinosort, (1, 1, size))  # [nproj, nz, ni]
    list1 = cp.mean(sinosort[ndrop : nproj - ndrop], axis=0)  # [nz, ni]
    del sinosort
    list2 = cp.mean(sinosmooth[ndrop : nproj - ndrop], axis=0)  # [nz, ni]
    listfact = list1 / list2  # [nz, ni]
    listmask = _detect_stripe_batch(listfact, snr)  # [nz, ni]
    listmask = binary_dilation(
        listmask, iterations=1, structure=cp.ones((1, 3), dtype=bool)
    ).astype(listmask.dtype)
    if norm:
        tomo = tomo / listfact[None]
    sinosmooth_t = cp.ascontiguousarray(
        cp.transpose(sinosmooth, (2, 1, 0))
    )  # [ni, nz, nproj]
    del sinosmooth
    sino_corrected = cp.transpose(_unsort3(ids, sinosmooth_t), (2, 1, 0))
    del sinosmooth_t, ids
    cp.copyto(tomo, sino_corrected, where=(listmask[None] > 0.0))
    return tomo


def _rs_dead3(tomo, snr, size, norm=True):
    """Batched _rs_dead for 3D input [nproj, nz, ni]."""
    tomo = cp.copy(tomo)
    nproj, nz, ni = tomo.shape
    sinosmooth = uniform_filter1d(tomo, 10, axis=0)
    listdiff = cp.sum(cp.abs(tomo - sinosmooth), axis=0)  # [nz, ni]
    listdiffbck = median_filter(listdiff, (1, size))  # [nz, ni]
    listfact = listdiff / listdiffbck  # [nz, ni]
    listmask = _detect_stripe_batch(listfact, snr)  # [nz, ni]
    listmask = binary_dilation(
        listmask, iterations=1, structure=cp.ones((1, 3), dtype=bool)
    ).astype(listmask.dtype)
    listmask[:, 0:2] = 0.0
    listmask[:, -2:] = 0.0
    for m in range(nz):
        listx = cp.where(listmask[m] < 1.0)[0]
        listxmiss = cp.where(listmask[m] > 0.0)[0]
        if len(listxmiss) > 0:
            matz = tomo[:, m, listx]
            ids = cp.searchsorted(listx, listxmiss)
            tomo[:, m, listxmiss] = matz[:, ids - 1] + (listxmiss - listx[ids - 1]) * (
                matz[:, ids] - matz[:, ids - 1]
            ) / (listx[ids] - listx[ids - 1])
    if norm:
        tomo = _rs_large3(tomo, snr, size)
    return tomo


def remove_all_stripe(tomo, snr=3, la_size=61, sm_size=21, dim=1):
    """
    Remove all types of stripe artifacts from sinogram using Nghia Vo's
    approach :cite:`Vo:18` (combination of algorithm 3,4,5, and 6).

    Parameters
    ----------
    tomo : ndarray
        3D tomographic data.
    snr  : float
        Ratio used to locate large stripes.
        Greater is less sensitive.
    la_size : int
        Window size of the median filter to remove large stripes.
    sm_size : int
        Window size of the median filter to remove small-to-medium stripes.
    dim : {1, 2}, optional
        Dimension of the window.

    Returns
    -------
    ndarray
        Corrected 3D tomographic data.
    """
    tomo = _rs_dead3(tomo, snr, la_size)
    tomo = _rs_sort3(tomo, sm_size, dim)
    return tomo
