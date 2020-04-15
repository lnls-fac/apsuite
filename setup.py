#!/usr/bin/env python-sirius

from setuptools import setup, find_packages


with open('VERSION', 'r') as _f:
    __version__ = _f.read().strip()

setup(
    name='apsuite',
    version=__version__,
    author='lnls-fac',
    description='High level Accelerator Physics functions',
    url='https://github.com/lnls-fac/apsuite',
    download_url='https://github.com/lnls-fac/apsuite',
    license='MIT License',
    classifiers=[
        'Intended Audience :: Science/Research',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering'
    ],
    packages=find_packages(),
    package_data={'apsuite': ['VERSION']},
    zip_safe=False)
