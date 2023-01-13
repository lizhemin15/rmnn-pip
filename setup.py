import setuptools

with open("README.md", "r") as fh:
  long_description = fh.read()

setuptools.setup(
  name="rmnn",
  version="0.0.1",
  author="Zhemin Li",
  author_email="lizhemin15@163.com",
  description="A small package for represent tensors or matrix with neural network based on Pytorch",
  long_description=long_description,
  long_description_content_type="text/markdown",
  url="https://gitee.com/lizhemin15/rmnn-pip",
  packages=setuptools.find_packages(),
  classifiers=[
  "Programming Language :: Python :: 3",
  "License :: OSI Approved :: MIT License",
  "Operating System :: OS Independent",
  ],
)