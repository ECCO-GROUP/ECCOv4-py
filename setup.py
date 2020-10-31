from setuptools import setup

def README():
    with open('README.md') as f:
        return f.read()

setup(
  name = 'ecco_v4_py',
  packages = ['ecco_v4_py'], # this must be the same as the name above
  version = '1.4.0',
  description = 'Estimating the Circulation and Climate of the Ocean (ECCO) Version 4 Python Package',
  author = 'Ian Fenty',
  author_email = 'ian.fenty@jpl.nasa.gov',
  url = 'https://github.com/ECCO-GROUP/ECCOv4-py',
  keywords = ['ecco','climate','mitgcm','estimate','circulation','climate'],
  include_package_data=True,
  data_files=[('binary_data',['binary_data/basins.data', 'binary_data/basins.meta'])],
  install_requires=[
	'dask[complete]',
	'python-dateutil',
	'matplotlib',
	'numpy',
	'pyresample',
	'xarray',
	'xmitgcm',
	'cartopy',
	'xgcm>=0.5.0',
        'future',
        'pathlib'],
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
