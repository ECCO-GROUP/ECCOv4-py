import setuptools

from distutils.core import setup

setup(
  name = 'ecco_v4_py',
  packages = ['ecco_v4_py'], # this must be the same as the name above
  version = '1.1.9',
  description = 'Estimating the Circulation and Climate of the Ocean (ECCO) Version 4 Python Package',
  author = 'Ian Fenty',
  author_email = 'ian.fenty@jpl.nasa.gov',
  url = 'https://github.com/ECCO-GROUP/ECCOv4-py',
  keywords = ['ecco','climate','mitgcm','estimate','circulation','climate'],
  include_package_data=True,
  data_files=[('binary_data',['binary_data/basins.data', 'binary_data/basins.meta'])],
  install_requires=[
	'cython',
	'shapely',
	'proj',
	'six',
	'dask[complete]',
	'datetime',
	'python-dateutil',
	'matplotlib',
	'numpy',
	'pyresample',
	'xarray',
	'xmitgcm',
	'pyyaml',
	'pyproj',
	'pykdtree',
	'cartopy',
 'cmocean',
	'xgcm'],
  classifiers=[
      'Development Status :: 5 - Production/Stable',
      'Intended Audience :: Science/Research', 
      'License :: OSI Approved :: MIT License',
      'Natural Language :: English',
      'Programming Language :: Python',
      'Programming Language :: Python :: 3.7',
      'Topic :: Scientific/Engineering :: Physics'
  ]
)
