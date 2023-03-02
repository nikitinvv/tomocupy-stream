=======
Install
=======

1. Create environment with necessary dependencies

::

    (base)$ conda create -n tomocupy_stream -c conda-forge cupy scikit-build swig pywavelets numexpr opencv tifffile h5py python=3.9


.. warning:: Conda has a built-in mechanism to determine and install the latest version of cudatoolkit supported by your driver. However, if for any reason you need to force-install a particular CUDA version (say 11.0), you can do:

::

    $ conda install -c conda-forge cupy cudatoolkit=11.0

2. Activate tomocupy_stream environment

::

    (base)$ conda activate tomocupy_stream

3. Install pytorch

::

    (tomocupy_stream)$ pip install torch torchvision torchaudio 


4. Install the pytorch pywavelets package for ring removal

::

    (tomocupy_stream)$ git clone https://github.com/fbcotter/pytorch_wavelets
    (tomocupy_stream)$ cd pytorch_wavelets
    (tomocupy_stream)$ pip install .
    (tomocupy_stream)$ cd -

5. Intall meta for supporting hdf meta data writer used by option: --save-format h5

::

    (tomocupy_stream)$ git clone https://github.com/xray-imaging/meta.git
    (tomocupy_stream)$ cd meta
    (tomocupy_stream)$ pip install .
    (tomocupy_stream)$ cd -


6. Make sure that the path to nvcc compiler is set (or set it by e.g. 'export CUDACXX=/local/cuda-11.7/bin/nvcc') and install tomocupy_stream

::
    
    (tomocupy_stream)$ git clone https://github.com/tomography/tomocupy_stream
    (tomocupy_stream)$ cd tomocupy_stream
    (tomocupy_stream)$ pip install .

==========
Unit tests
==========
Check the library path to cuda or set it by 'export LD_LIBRARY_PATH=/local/cuda-11.7/lib64'

Run the following to check all functionality
::

    (tomocupy_stream)$ cd tests; bash test_all.sh


Update
======

**tomocupy_stream** is constantly updated to include new features. To update your locally installed version

::

    (tomocupy_stream)$ cd tomocupy_stream
    (tomocupy_stream)$ git pull
    (tomocupy_stream)$ pip install .
