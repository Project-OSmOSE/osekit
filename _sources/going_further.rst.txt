
=================
Going further 
=================

Installation with github and poetry
===================================

.. note::

	We have made a :ref:`tutorial on Github <Github>` if you are starting with it


In this section we give you the instructions to install our toolkit from github using the `poetry <https://python-poetry.org/>`_ tool:


1. First clone our Github project repository

.. code:: bash

	git clone https://github.com/Project-OSmOSE/osmose-toolkit.git


2. Create and activate a Conda virtual environment in python 3.10 with the name you want (here osmose_dev)

.. code:: bash

	conda create --name osmose_dev python=3.10 -y
	conda activate osmose_dev

3. Install poetry and use it to install our toolkit

.. code:: bash

	conda install poetry
	poetry install

.. note::
	`poetry install` will use the file `pyproject.toml` so be sure to execute this command from its directory. Besides, if `poetry install` fails with the message ``ImportError: cannot import name 'PyProjectException'``, run this line instead:

	.. code:: bash

		conda install -c conda-forge poetry==1.3.2


The toolkit is now installed in editable mode! You can call it just like any regular python package as long as you are in its conda environment

.. code:: python

	import OSmOSE as osm
	dataset = osm.Dataset()


Note that on the contrary of the installation procedure given in :ref:`Getting started`, the OSmOSE toolkit is now installed in editable mode, meaning that any change made to the package's file will be reflected immediately on the environment, without needing to reload it. 






Contribute
=================


Here is the general workflow for contributing to our project:


1. Install locally our toolkit (see section :ref:`Installation with github and poetry` above);

2. Start by reviewing our Github `issues <https://github.com/Project-OSmOSE/osmose-toolkit/issues>`_ and propose new issues anytime following standard procedures (in particular, among other must-have: short description, assignees, status, label...);

3. Develop and validate locally your contribution. Please follow standad github procedures such as developing on a new branch built from our main branch; no pushes will be accepted on our main repository ;

4. (Optional) If intended to be deployed on Datarmor, you will have to also validate your contribution on it (see section :ref:`On Datarmor` for details);

5. Once validated, commit and push your branch to our Github project and create a pull-request so it can be properly peer-reviewed. 

6. Change the version in pyproject.toml depending on the importance of the changes: 
	- small patch : change patch version (x.x.X)
	- non breaking changes : change minor version (x.X.x)
	- breaking changes : change major version (X.x.x)

7. Use `poetry build` and upload generated files (present locally in the folder `~/dist`) in the upgraded package version on github `here <https://github.com/Project-OSmOSE/osmose-toolkit/releases/tag/v0.1.0>`_ 

.. include:: gallery_advanced_workflows/index.rst




