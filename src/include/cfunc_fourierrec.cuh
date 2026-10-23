#ifndef CFUNC_FOURIERREC_CUH
#define CFUNC_FOURIERREC_CUH

#include <cufft.h>
#include <cufftXt.h>
#include <cuda_fp16.h>
#include "defs.cuh"


class cfunc_fourierrec {
  bool is_free = false;

  size_t m;
  float mu;

  
  float *x;
  float *y;
  float* theta;
  real2 *fde;
  real2 *ge;

  cufftHandle plan2d;
  cufftHandle plan1d;
  cufftHandle plan_filter_fwd;
  cufftHandle plan_filter_inv;

  // binned-scatter index (see build_bins), rebuilt only when theta changes
  int tile_b = 32;              // tile side on the padded grid
  int subsize = 4096;           // max samples per subproblem
  int ntx = 0;                  // tiles per axis
  int nsub = 0;                 // number of subproblems
  int nscratch = 0;             // tiles split over several subproblems
  size_t shmem = 0;
  size_t scratch_bytes = 0;
  int *bin_samples = nullptr;
  int *sub_tile = nullptr;
  int *sub_beg = nullptr;
  int *sub_end = nullptr;
  int *sub_scratch = nullptr;
  int *fin_tile = nullptr;
  float2 *scratch = nullptr;

  float *theta_host = nullptr;  // pinned, for the theta-change check
  float *theta_cached = nullptr;
  size_t nproj_cached = 0;

  void build_bins(size_t nproj0, cudaStream_t stream);
  void free_bins();

public:
  size_t n;      // width of square slices
  size_t nproj; // number of angles
  size_t nz;    // number of slices
  size_t ne;
  cfunc_fourierrec(size_t nproj, size_t nz, size_t n);
  ~cfunc_fourierrec();
  void backprojection(size_t f, size_t g, size_t theta_, size_t nproj0, size_t stream);  
  void free();
};

#endif