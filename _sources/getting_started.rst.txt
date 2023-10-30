
=================
Getting started
=================

 

Quick installation
===================

The quickest and safest way to start using our toolkit is to install its latest stable version locally in a virtual environment. To do so follow these steps:

1. Download the `wheel file <https://github.com/Project-OSmOSE/osmose-toolkit/releases/tag/v0.1.0>`_ (.whl extension) of the latest version available in your local working folder

2. Create and activate a Conda virtual environment in python 3.10, named for example osmose_stable

.. code-block:: bash	

	conda create --name osmose python=3.10 -y
	conda activate osmose

.. note::

	We have made a :ref:`tutorial on Conda <Conda>` if you are starting with it


3. Install the pip tool in your conda environment, use it to install the toolkit from the wheel file and delete this file

.. code-block:: bash

	conda install pip
	pip install <toolkit_version.whl>
        rm <toolkit_version.whl>

.. note:: 
	If it happens that the toolkit version has not changed, use instead

	.. code-block:: bash

		pip install --force-reinstall <toolkit_version.whl>



.. warning::
	We discourage users to install the toolkit from our git repository, as it is under permanent development and thus potentially unstable and unfinished.

.. note::
	If for any reason you get an unexpected error message or an incorrect result, please open a new issue in the `issue tracker <https://github.com/Project-OSmOSE/osmose-toolkit/issues>`_ 



Sample datasets
=============================

Our toolkit comes with a diversified set of sample datasets containing both audio recordings and auxiliary data, described in the table below.

======================= ===================== ========================= ======================================================= ======================= =====================================================
Dataset name               Time period              Location                  Predominant sources                                  Sample frequency (Hz)    Audio recorder
======================= ===================== ========================= ======================================================= ======================= =====================================================
SPM2011                 2013--  -> 2013        Saint Pierre et Miquelon                                                                                   Fixed hydrophone
BallenyIslands2015      20150115 -> 20160110        Southern Ocean               Blue whales, seisms                                       1000                     Fixed hydrophone
SES                     2019-04-12 ->          Southern Indian Ocean      Wind speed, flow noise                                    50000                  Bio-logged elephant seal
======================= ===================== ========================= ======================================================= ======================= =====================================================

Most of the examples found in modules and in the use case gallery use these audio recordings. To test and run the scripts, we recommend downloading these files to your local hard drive. The files can be downloaded from the GitHub `repository <https://github.com/project-OSmOSE>`_


  


.. include:: gallery_tutorials/index.rst


.. include:: tutorials.rst



