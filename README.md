# mac-torch-bench

A simple set of test to benchmark Apple M1. It's not meant to produce useful models! It's meant to push the hardware to see how it performs under different loads. Does more memory and GPU matter and how much?

Thanks to sgrvinod example on github. I borrowed code from his example, but had to tweak it heavily to get CPU to run on MacOS. Still trying to get it to run on M2Max GPU.

[https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection] pytorch tutorial by sgrvinod

# Install Conda

The benchmark uses conda to create an environment and installs the necessary tensorflow packages. You can download miniconda package and install it.

## MacOS

conda env update -f environment.yml

## Windows

conda env create -f environment-win.yml

[https://docs.conda.io/en/latest/miniconda.html#macos-installers] installers

Select ARM 64 pkg and run the installer.

# Setup with Conda

1. git clone https://github.com/woolfel/mac-torch-bench
2. cd mac-torch-bench
3. conda env update environment.yml
4. conda activate torch_bench
5. verify everything is installed correctly by running env_check.py