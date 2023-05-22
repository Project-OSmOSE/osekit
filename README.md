[![Continuous integration][ci-badge]][ci-link]
[![Code style: black][black-badge]][black-link]
[![][coverage-badge]][coverage-link]

[ci-badge]: https://github.com/Project-ODE/osmose-toolkit/actions/workflows/github-ci.yml/badge.svg
[ci-link]: https://github.com/Project-ODE/osmose-toolkit/actions/workflows/github-ci.yml
[black-badge]: https://img.shields.io/badge/code%20style-black-000000.svg
[black-link]: https://github.com/psf/black

# OSmOSE_package


## Installation

OSmOSE is currently in development, and no release exist yet. Thus, installing the package means using an unstable, unfinished version.

### Local installation

Follow this step if you are installing the package from scratch locally on your computer. To use on Datarmor, skip to the **Use on DATARMOR** section.

1. First, create an empty git repository in a newly created folder and clone this repository.

```bash
git init

git clone https://github.com/Project-ODE/OSmOSE_package.git
```

2. Create a Conda virtual environment in python 3.10.

```bash
conda create --name osmose python=3.10 -y
conda activate osmose
```

3. Install poetry and use it to install the package.

```bash
conda install poetry
poetry install
```

Note: if `poetry install` fails with the message ``ImportError: cannot import name 'PyProjectException'``, run this line instead:

```bash
conda install -c conda-forge poetry==1.3.2
```

The package is installed in editable mode! You can call it just like any regular python package as long as you are in the environment.

```python
import OSmOSE as osm

dataset = osm.Dataset()
```

Note that it is installed in editable mode, meaning that any change made to the package's file will be reflected immediately on the environment, without needing to reload it. 

### Use on DATARMOR

If you are on DATARMOR, then the initial setup is already done! The conda environment is named osmose_dev and the package can be found on `/home/datawork-osmose/osmose_package/`. If you are using Jupyter Hub, then you can just change the kernel to `osmose_dev` and start using the OSmOSE package. Note that without the command line, you will not automatically update the package and might be missing a newer version.

On the command line, run this line just once:

```csh
echo "alias osmose_activate='cd /home/datawork-osmose/osmose_package; git checkout main; git pull origin main; . /appli/anaconda/latest/etc/profile.d/conda.sh; conda activate /home/datawork-osmose/conda-env/osmose_dev/; cd -'" >> .bashrc
```

Then reload the shell. From now on, typing 
```bash
bash
osmose_activate
``` 
will:

- Update the local package to the latest version.

- Activate the conda environment.

- Update or install any dependency that is not present.

Note that the environment is a development environment and might not be suitable for production.