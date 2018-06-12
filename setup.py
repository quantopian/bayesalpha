#!/usr/bin/env python
from os.path import realpath, dirname, join
from setuptools import setup, find_packages
import versioneer

DISTNAME = 'bayesalpha'
AUTHOR = 'Adrian Seyboldt'
AUTHOR_EMAIL = 'adrian.seyboldt@gmail.com'

requirements = [
    'Bottleneck>=1.1',
    'pymc3>=3.4.1',
    'scipy>=0.19.0',
    'xarray>=0.9',
    'seaborn',
    'empyrical',
]


if __name__ == "__main__":
    setup(
        name=DISTNAME,
        version=versioneer.get_version(),
        cmdclass=versioneer.get_cmdclass(),
        packages=find_packages(),
        install_requires=requirements,
        test_requires=['pytest']
    )
