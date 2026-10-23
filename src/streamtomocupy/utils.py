import cupy as cp

# Normalize in float32 and cast once at the end: raw uint16 counts do not
# survive a float16 cast (>=65520 overflows to inf, and the ulp at 60000 is 32).
darkflat_kernel = cp.ElementwiseKernel(
    "T data, float32 dark0, float32 scale",
    "U out",
    "out = (U)(((float)data - dark0) * scale);",
    "darkflat",
)

# Same reason: take the log in float32, store in the working dtype.
minus_log_kernel = cp.ElementwiseKernel(
    "T inp",
    "T out",
    """
    float v = (float)inp;
    v = v <= 0 ? 0.0f : -logf(v);
    if (isnan(v)) v = 6.0f;
    if (isinf(v)) v = 0.0f;
    out = (T)v;
    """,
    "minus_log",
)
