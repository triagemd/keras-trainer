#!/usr/bin/env python

from setuptools import setup, find_packages


setup(
    name='keras-trainer',
    version='0.0.17',
    description='A training abstraction for Keras models.',
    author='Triage Technologies Inc.',
    author_email='ai@triage.com',
    url='https://www.triage.com/',
    packages=find_packages(exclude=['tests', '.cache', '.venv', '.git', 'dist']),
    install_requires=[
        'Keras',
        'h5py',
        'Pillow',
        'keras-model-specs>=0.0.16',
    ]
)
