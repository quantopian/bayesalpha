#!/usr/bin/env python
from os.path import realpath, dirname, join
from setuptools import setup, find_packages

DISTNAME = 'bayesalpha'
AUTHOR = 'Adrian Seyboldt'
AUTHOR_EMAIL = 'adrian.seyboldt@gmail.com'
VERSION = '0.1'

PROJECT_ROOT = dirname(realpath(__file__))
REQUIREMENTS_FILE = join(PROJECT_ROOT, 'requirements.txt')

with open(REQUIREMENTS_FILE) as reqfile:
    install_reqs = reqfile.read().splitlines()

if __name__ == "__main__":
    setup(name=DISTNAME,
          version=VERSION,
          packages=find_packages(),
          install_requires=install_reqs)
