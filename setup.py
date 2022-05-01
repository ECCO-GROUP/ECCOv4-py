from setuptools import setup

def README():
    with open('README.md') as f:
        return f.read()

setup(
  name = 'ecco_v4_py',
  packages = ['ecco_v4_py'], # this must be the same as the name above
  version = '1.5.3',
  description = 'Estimating the Circulation and Climate of the Ocean (ECCO) Version 4 Python Package',
  author = 'Ian Fenty, Ou Wang, Tim Smith, and others',
  author_email = 'ian.fenty@jpl.nasa.gov, ecco-group@mit.edu',
  url = 'https://github.com/ECCO-GROUP/ECCOv4-py',
  keywords = ['ecco','climate','mitgcm','estimate','circulation','climate'],
  include_package_data=True,
  data_files=[('binary_data',['binary_data/basins.data', 'binary_data/basins.meta'])],
  python_requires = '>=3.7',
  install_requires=[
	'cartopy>=0.18.0',
    'cmocean',
	'dask[complete]',
    'future',
	'geos',
	'matplotlib',
    'netcdf4',
	'numpy >= 1.17',
    'proj',
	'pyresample',
	'python-dateutil',
	'xarray',
	'xmitgcm',
	'xgcm=0.6.1'],
  tests_require=['pytest','coverage'],
  license='MIT',
  classifiers=[
      'Development Status :: 5 - Production/Stable',
      'Intended Audience :: Science/Research',
      'License :: OSI Approved :: MIT License',
      'Natural Language :: English',
      'Programming Language :: Python',
      'Programming Language :: Python :: 3.7',
      'Topic :: Scientific/Engineering :: Physics'
  ],
  long_description=README(),
  long_description_content_type='text/markdown'
)
