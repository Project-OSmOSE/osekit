


Tutorials
=================


Github
------------

If you happen to encounter this error "fatal: unable to access 'https://github.com/Project-OSmOSE/osmose-toolkit.git/': The requested URL returned error: 403" after running `git push --force origin bugfix`, here is how you can deal with it 

.. code-block:: bash

	git remote remove origin
	git remote add origin git@github.com:Project-OSmOSE/osmose-toolkit.git






Conda 
------------

To activate a conda environments after its creation

.. code-block:: bash

	conda activate ENV_NAME

To install a package in your activated environmnent

.. code-block:: bash

	conda install PACKAGE_NAME

As an alternative way of installing python packages in your conda environment, you can also run `conda install pip` and then run `pip install PACKAGE_NAME`. Note that some packages (eg `pysoundfile <https://pysoundfile.readthedocs.io/en/latest/>`_) are not available from conda install but only pip install (problem unsolved of `Solving environment: failed with initial frozen solve. Retrying with flexible solve.`)


To delete a conda environment, run 

.. code-block:: bash

	conda remove --name ENV_NAME


To get a list of all existing conda environments on your machine

.. code-block:: bash

	conda info --envs




Bash and qsub 
------------------

- `ls` : list all files and folders ;

- `qstat -u myusername`  : this will list all your launched qsub jobs. This useful to check that your processing is running as expected, for example there is the right number of running jobs. The letter S stands for job status, which can be R = running, Q = en attente (dans la queue), H = en attente (on hold) ;

- `cat some_file.txt` : this will display the content of a text file (.txt, .csv) in your terminal.

- `qdel job_id` : this is to kill a launched job appearing in the qstat list. For example to kill the second job from the list below run `qdel 1767379`


    r2i1n25 datahome/dcazau% qstat -u dcazau
    datarmor0:
                                                                Req'd  Req'd   Elap
    Job ID      	Username Queue	Jobname	SessID NDS TSK Memory Time  S Time
    --------------- -------- -------- ---------- ------ --- --- ------ ----- - -----
    1767039.datarmo dcazau   ice_mt   jupyterhub  42970   1   4  8192m 02:00 R 00:12
    1767379.datarmo dcazau   ice_mt   jupyterhub  34170   1   8   16gb 02:00 R 01:03


- `qstat -fx id_job` : this will show you all information on your job, especially the address of servers on which it is or will run. Example : exec_vnode = (r2i1n25:ncpus=8:mem=16777216kb) informs you that your code is running on single machine located at r2i1n25 over 8 CPUS with 16 GB RAM





