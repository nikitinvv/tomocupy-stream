========
StreamTomocuPy
========

**StreamTomocuPy** is a Python package designed for GPU-accelerated reconstruction of tomographic data, leveraging CUDA streams and multi-GPU setups. It supports both 16-bit and 32-bit precision for efficient, high-performance processing.

While it inherits the core functionality of **TomocuPy**, **StreamTomocuPy** is specifically optimized for fast, streaming data processing, minimizing I/O operations with disk storage. The package’s streaming capabilities and multi-GPU parallelization are abstracted away using a decorator function, making it easy to adjust the code and add new processing functions as needed.

The key functionality is encapsulated in the *StreamRecon* class, located in the *streamrecon.py* module. This class is instantiated once and reused to process data of the same size, or subsets of the data. During the initial setup, the class allocates GPU and pinned memory buffers, output arrays, and initializes necessary methods. This setup may take approximately 10 seconds.

To test the functionality see *tests/test.ipynb*.

================
Installation
================


~~~~~~
Check CUDA path
~~~~~~
Example

::

   export CUDA_HOME=/usr/local/cuda-12.1
   export PATH=${CUDA_HOME}/bin:${PATH}
   export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH

~~~~~~
Install necessary packages
~~~~~~

::

  conda create -n streamtomocupy cupy scikit-build swig cmake h5py pywavelets matplotlib notebook
  
  conda activate streamtomocupy

~~~~~~
Install streamtomocupy
~~~~~~

::
  
  git clone https://github.com/nikitinvv/streamtomocupy
  
  cd streamtomocupy
  
  pip install .
  
================
Tests
================

See tests/test.ipynb. Reconstruction parameters are set in tests.conf file.

================
Performance
================

Measured with *tests/bench.py* on 4x NVIDIA A100-SXM4-40GB (tomo5, 32 cores, 4 TB RAM). Each run is a single *StreamRecon.rec* call over an analytic phantom generated in RAM, so no disk I/O is included. The datasets are cubic, *nproj = nz = n*, with uint16 projections on input.

*ncz* is *nsino_per_chunk* = *nproj_per_chunk*. The 8192^3 rows use *ncz=4* because *ncz=8* does not fit in 40 GB, so they pay twice the per-chunk overhead and are not on the same scaling curve as the smaller volumes. *Setup (s)* is the one-time *StreamRecon* construction (buffer allocation and filter/interpolation tables); it is independent of the number of reconstructions that follow.

::

   python bench.py --n 4096 --dtype float16 --algorithm lprec \
          --nsino-per-chunk 8 --nproj-per-chunk 8 --mode rec --ngpus 4

fourierrec, float32
~~~~~~~~~~~~~~~~~~~

======  ===  =========  =========  ========  ========  =========
Volume  ncz  Setup (s)  Recon (s)  Gvoxel/s  GPU (GB)  Host (GB)
======  ===  =========  =========  ========  ========  =========
2048^3  8    2.93       2.53       3.4       3.7       51.0
4096^3  8    6.39       17.41      3.95      15.2      393.1
8192^3  4    11.76      151.28     3.63      30.4      3089.3
======  ===  =========  =========  ========  ========  =========

fourierrec, float16
~~~~~~~~~~~~~~~~~~~

======  ===  =========  =========  ========  ========  =========
Volume  ncz  Setup (s)  Recon (s)  Gvoxel/s  GPU (GB)  Host (GB)
======  ===  =========  =========  ========  ========  =========
2048^3  8    3.02       2.94       2.92      2.1       35.1
4096^3  8    6.35       14.68      4.68      9.0       265.1
8192^3  4    11.82      114.86     4.79      19.9      2065.2
======  ===  =========  =========  ========  ========  =========

lprec, float32
~~~~~~~~~~~~~~

======  ===  =========  =========  ========  ========  =========
Volume  ncz  Setup (s)  Recon (s)  Gvoxel/s  GPU (GB)  Host (GB)
======  ===  =========  =========  ========  ========  =========
2048^3  8    4.25       2.32       3.7       4.1       51.1
4096^3  8    6.93       16.79      4.09      17.0      393.1
8192^3  4    12.88      142.9      3.85      36.6      3089.3
======  ===  =========  =========  ========  ========  =========

lprec, float16
~~~~~~~~~~~~~~

======  ===  =========  =========  ========  ========  =========
Volume  ncz  Setup (s)  Recon (s)  Gvoxel/s  GPU (GB)  Host (GB)
======  ===  =========  =========  ========  ========  =========
2048^3  8    3.52       2.0        4.29      2.5       35.0
4096^3  8    7.03       14.03      4.9       10.5      265.1
8192^3  4    13.07      119.09     4.62      23.6      2065.3
======  ===  =========  =========  ========  ========  =========

*float16* is the faster choice at every volume except 2048^3 with *fourierrec*, where the reconstruction is short enough that the fixed costs dominate. It also roughly halves both the GPU and the host footprint. *lprec* in *float32* at 8192^3 is the tightest configuration in the table at 36.6 GB of the 40 GB available.

*Host (GB)* is peak resident set size, dominated by the input projections plus the output volume; it is what limits the volume size on a given machine, not the GPU.

================
Streaming data over RDMA
================

*tests/rdma* reconstructs a dataset that is never written to disk. The projections are built in the detector host's RAM and pulled across InfiniBand with one-sided RDMA READ, straight into the numpy array that *StreamRecon* chunks.

::

   cd tests/rdma

   ./run_demo.sh [nproj nz n]      # defaults 1024 512 2048 = 2.16 GB

*rdma.py* is a ctypes wrapper over *rdma_transport.c* and moves any C-contiguous numpy array, so there is nothing tomography-specific in the transport:

::

   import rdma

   rdma.serve(buf)                 # expose buf, block until the peer has pulled it

   secs = rdma.fetch(ip, buf)      # pull into buf, return the transfer time

Allocating one flat buffer and taking *proj*/*dark*/*flat* views of it (see *tomo_data.allocate*) keeps the dataset contiguous, so the transfer is a single memory registration and one stream of reads. Host names and the IPoIB address of the server are set at the top of *run_demo.sh*.

On a 100 Gb/s EDR link the default 2.16 GB dataset moves in 0.20 s, i.e. 10.8 GB/s, about 86% of the 12.5 GB/s line rate.


================
Profiling with Nvidia Nsys
================
4x Tesla A100 HBM memory. Parameters args.nsino_per_chunk=16, args.nproj_per_chunk=16, args.reconstruction_algorithm='lprec', args.dtype='float16'

::

   nsys profile python test_2k.py

Opened with nsys-ui

.. image:: nsys.png
   :alt: Nsys
   :width: 1200px
 


Example of output

.. image:: output.png
   :alt: output
   :width: 600px
