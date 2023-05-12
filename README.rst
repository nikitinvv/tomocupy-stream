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
  
  pip install torch torchvision torchaudio
  
  git clone https://github.com/fbcotter/pytorch_wavelets
  
  cd pytorch_wavelets; pip install .; cd -  
  
  git clone https://github.com/nikitinvv/tomocupy-stream
  
  cd tomocupy-stream
  
  pip install .
  
