from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='pykifmm2d',
    version='0.0.1',
    description='Pure python Kernel-Independent FMM in 2D',
    long_description=long_description,
    url='https://github.com/dbstein/pykifmm2d/',
    author='David Stein',
    author_email='dstein@flatironinstitute.org',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Scientists/Mathematicians',
        'License :: Apache 2',
        'Programming Language :: Python :: 3',
    ],
    packages=find_packages(),
    install_requires=[],
)
