from setuptools import setup

setup(name='stardate_code',
      version='0.0.1',
      description='Inferring stellar ages using stellar evolution models and gyrochronology',
      url='http://github.com/RuthAngus/stardate',
      author='Ruth Angus',
      author_email='ruthangus@gmail.com',
      license='MIT',
      packages=['stardate'],
      install_requires=['numpy', 'pandas', 'h5py', 'tqdm', 'isochrones', 'emcee'],
      zip_safe=False)
