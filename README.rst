========
Tomocupy_stream
========

**Tomocupy_stream** is a Python package for GPU reconstruction of tomographic data in 16-bit and 32-bit precision. All preprocessing operations are implemented on GPU with using CuPy library, the backprojection operation is implemented with CUDA C.


================
Installation
================

~~~~~~
Install necessary packages
~~~~~~

::

  conda create -n tomocupy -c conda-forge cupy scikit-build swig pywavelets numexpr opencv tifffile h5py cupy cudatoolkit=11.0 python=3.9
  
  conda activate tomocupy

~~~~~~
Install pytorch for wavelet filtering
~~~~~~

::


  pip install torch torchvision torchaudio
  
  git clone https://github.com/fbcotter/pytorch_wavelets
  
  cd pytorch_wavelets; pip install .; cd -  


~~~~~~
Install jupyter notebook 
~~~~~~

::

  pip install jupyter

~~~~~~
Install tomocupy-stream
~~~~~~

::
  
  git clone https://github.com/nikitinvv/tomocupy-stream
  
  cd tomocupy-stream
  
  pip install .
  
================
Tests
================


~~~~~~
Demonstration of reconstruction and reprojection in jupyter notebook
~~~~~~

https://github.com/nikitinvv/tomocupy-stream/blob/main/tests/test_for_compression.ipynb

~~~~~~
Adjoint test
~~~~~~
tests/test_chip_adjoint.py

~~~~~~
Reconstruction from h5 file
~~~~~~
tests/test_rec.py

~~~~~~
Performance test
~~~~~~
tests/test_perf.py




