# pose_analysis

[![Installation Test](https://github.com/edeno/pose_analysis/actions/workflows/install-test.yml/badge.svg)](https://github.com/edeno/pose_analysis/actions/workflows/install-test.yml)

### Installation ###

1. Install miniconda (or anaconda) if it isn't already installed. Type into bash (or install from the anaconda website):

```bash
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh;
bash miniconda.sh -b -p $HOME/miniconda
export PATH="$HOME/miniconda/bin:$PATH"
hash -r
```

2. Enter the `pose_analysis` folder on your computer (```cd .../pose_analysis```) and install the anaconda environment for the repository. Type into bash:

```bash
conda env create -f environment.yml # create a conda environment
conda activate pose_analysis # activate conda environment
python setup.py develop # installs src package and allows editing
```
