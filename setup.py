""" setup.py """
import os
from setuptools import setup, find_packages

def get_description():
    """ Makes README file into a string"""
    with open("README.md") as file:
        long_description = file.read()
    return long_description

def parse_requirements(filename):
    """ load requirements from a pip requirements file """
    lineiter = (line.strip() for line in open(filename))
    return [line for line in lineiter if line and not line.startswith("#")]

def get_version():
    """ Gets version from rmnn.__init__.py

    Runs `rmnn.__init__` and loads defined variables into scope
    """
    with open(os.path.join('rmnn', '__init__.py')) as version_file:
        # pylint: disable=exec-used, undefined-variable
        exec(version_file.read(), globals())
        return __version__

setup(
    name='rmnn',
    author='Zhemin Li',
    author_email='lizhemin15@163.com',
    version=get_version(),
    url="https://gitee.com/lizhemin15/rmnn-pip",
    download_url='https://gitee.com/lizhemin15/rmnn-pip',
    description='A small package for represent tensors or matrix with neural network based on Pytorch',
    long_description=get_description(),
    packages=find_packages(exclude=['docs']),
    install_requires=parse_requirements("requirements/common.txt"),
    keywords='matrix representation',
    classifiers=['Development Status :: 3 - Alpha',
                 'Intended Audience :: Science/Research',
                 'Intended Audience :: Developers',
                 'Programming Language :: Python',
                 'Topic :: Software Development',
                 'Topic :: Scientific/Engineering'],
    extras_require={
        'dev': ['pylint', 'sphinx'],
        'test': [],
    },
    license='GPL-3.0'
)


