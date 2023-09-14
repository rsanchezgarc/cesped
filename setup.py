"""A setuptools based setup module.

See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

with open(path.join(here, 'cesped', '__init__.py'), encoding='utf-8') as f:
    version = f.readline().split("=")[-1].strip().strip('"')

setup(
    name='cesped',
    version=version,
    description='Code utilities for the CESPED (Cryo-EM Supervised Pose Estimation Dataset) benchmark',
    long_description=long_description,  # Optional
    url='https://github.com/Anonymous/cesped',  # Optional
    author='Anonymous',  # Optional
    author_email='Anonymous',  # Optional
    keywords='deep learning cryoem pose estimation',  # Optional
    packages=find_packages(),
    install_requires=[requirements],
    include_package_data=True  # This line is important to read MANIFEST.in
)
