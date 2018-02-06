#!/bin/bash
set -e

export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python2.7
export WORKON_HOME=/mnt/jenkins_backups/virtual_envs
if [ ! -d $WORKON_HOME ]; then
  mkdir $WORKON_HOME
fi
source /usr/local/bin/virtualenvwrapper.sh

# Create virtualenv and install necessary packages
if ! workon bayesalpha; then
    mkvirtualenv bayesalpha
fi

pip install setuptools==36.6.0
pip install tox==2.9.1

# Args after the bare '--' are forwarded to py.test.
tox --recreate -- --junitxml="$(pwd)/pytest.xml"
