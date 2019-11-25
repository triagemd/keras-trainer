#!/usr/bin/env python

from setuptools import setup, find_packages


setup(
    name='keras-trainer',
    version='1.1.0',
    description='A training abstraction for Keras models.',
    author='Triage Technologies Inc.',
    author_email='ai@triage.com',
    url='https://www.triage.com/',
    packages=find_packages(exclude=['tests', '.cache', '.venv', '.git', 'dist']),
    install_requires=[
        'Keras==2.2.4',
        'h5py',
        'pandas',
        'Pillow',
        'keras-model-specs==1.2.0',
        'sklearn'
    ]
)
