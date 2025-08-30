from setuptools import setup

def README():
    with open('README.md') as f:
        return f.read()

setup(
  name = 'ecco_v4_py',
  packages = ['ecco_v4_py'], # this must be the same as the name above
  version = '1.7.8',
  description = 'Estimating the Circulation and Climate of the Ocean (ECCO) Version 4 Python Package',
  author = 'Ian Fenty, Ou Wang, Tim Smith, Andrew Delman, and others',
  author_email = 'ian.fenty@jpl.nasa.gov, ecco-group@mit.edu',
  url = 'https://github.com/ECCO-GROUP/ECCOv4-py',
  keywords = ['ecco','climate','mitgcm','estimate','circulation','climate'],
  include_package_data=True,
  data_files=[('binary_data',['binary_data/basins.data', 'binary_data/basins.meta'])],
  python_requires = '>=3.7',
  install_requires=[
  'aiobotocore==2.24.1',
  'numpy',
  'future',
  'bottleneck',
  'cartopy',
  'cmocean',
  'dask[complete]',
  'fsspec == 2025.7.0',
# 'fsspec >= 2024.12.0',
  'matplotlib',
  'netCDF4',
  'pandas',
  'pyresample',
  'python-dateutil',
  'requests',
  's3fs == 2025.7.0',
# 's3fs >= 2024.12.0',
  'tqdm',
  'xarray',
  'xmitgcm',
  'xgcm >= 0.5.0',
  'zarr >= 3.0.7'],
  tests_require=['pytest','coverage'],
  license='MIT',
  classifiers=[
      'Development Status :: 5 - Production/Stable',
      'Intended Audience :: Science/Research',
      'Natural Language :: English',
      'Programming Language :: Python',
      'Programming Language :: Python :: 3.7',
      'Programming Language :: Python :: 3.8',
      'Programming Language :: Python :: 3.9',
      'Programming Language :: Python :: 3.10',
      'Programming Language :: Python :: 3.11',
      'Programming Language :: Python :: 3.12',
      'Programming Language :: Python :: 3.13',
      'Topic :: Scientific/Engineering :: Physics'
  ],
  long_description=README(),
  long_description_content_type='text/markdown'
)
