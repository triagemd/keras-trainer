#!/usr/bin/env python

from setuptools import setup, find_packages


setup(
    name='keras-trainer',
    version='0.0.2',
    description='A training abstraction for Keras models.',
    author='Triage Technologies Inc.',
    author_email='ai@triage.com',
    url='https://www.triage.com/',
    packages=find_packages(exclude=['tests', '.cache', '.venv', '.git', 'dist']),
    install_requires=[
        'keras',
        'h5py',
        'Pillow',
        'ml-tools',
    ]
)
