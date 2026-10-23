import cupy as cp
import cupyx.scipy.ndimage as ndimage

from streamtomocupy import remove_stripe
from streamtomocupy import retrieve_phase
from streamtomocupy.utils import darkflat_kernel, minus_log_kernel


class Proc:
    def __init__(self, args):
        self.args = args

    def darkflat_correction(self, data, dark, flat, out=None):
        """Dark-flat field correction"""
        args = self.args
        # works only for processing all angles
        dark0 = cp.mean(dark, axis=0, dtype="float32")
        flat0 = cp.mean(flat, axis=0, dtype="float32")
        scale = 1 / (flat0 - dark0 + 1e-5)
        if out is None:
            out = cp.empty(data.shape, dtype=args.dtype)
        return darkflat_kernel(data, dark0, scale, out)

    def remove_outliers(self, data):
        """Remove outliers"""
        args = self.args

        if int(args.dezinger) > 0:
            w = int(args.dezinger)
            if len(data.shape) == 3:
                fdata = ndimage.median_filter(data, [w, 1, w])
            else:
                fdata = ndimage.median_filter(data, [w, w])
            data[:] = cp.where(
                cp.logical_and(data > fdata, (data - fdata) > args.dezinger_threshold),
                fdata,
                data,
            )
        return data

    def minus_log(self, data):
        """Taking negative logarithm"""
        args = self.args

        if args.minus_log == "True":
            minus_log_kernel(data, data)
        return data

    def remove_stripe(self, res):
        """Remove stripes"""
        args = self.args

        if args.remove_stripe_method == "fw":
            res[:] = remove_stripe.remove_stripe_fw(
                res, args.fw_sigma, args.fw_filter, args.fw_level
            )
        elif args.remove_stripe_method == "ti":
            res[:] = remove_stripe.remove_stripe_ti(res, args.ti_beta, args.ti_mask)
        elif args.remove_stripe_method == "vo-all":
            res[:] = remove_stripe.remove_all_stripe(
                res,
                args.vo_all_snr,
                args.vo_all_la_size,
                args.vo_all_sm_size,
                args.vo_all_dim,
            )
        return res

    def retrieve_phase(self, data):
        """Retrieve phase"""
        args = self.args

        if (
            args.retrieve_phase_method == "Gpaganin"
            or args.retrieve_phase_method == "paganin"
        ):
            data[:] = retrieve_phase.paganin_filter(
                data,
                args.pixel_size * 1e-4,
                args.propagation_distance / 10,
                args.energy,
                args.retrieve_phase_alpha,
                args.retrieve_phase_method,
                args.retrieve_phase_delta_beta,
                args.retrieve_phase_W * 1e-4,
            )  # units adjusted based on the tomopy implementation
        return data
