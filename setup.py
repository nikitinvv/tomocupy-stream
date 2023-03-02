from skbuild import setup
from setuptools import find_packages

setup(
    name='tomocupy_stream',
    version=open('VERSION').read().strip(),
    author='Viktor Nikitin',
    package_dir={"": "src"},    
    packages=find_packages('src'),
    zip_safe=False,
)
