import glob
import os
import json
import tomlkit
import subprocess
from string import Template
from typing import NamedTuple, List, Literal
from warnings import warn
from pathlib import Path
from datetime import datetime
from importlib import resources
from OSmOSE.utils import read_config, set_umask
from OSmOSE.config import *


class Job_builder:
    def __init__(self, config_file: str = None):
        if config_file is None:
            self.__configfile = "config.toml"
            self.__full_config: NamedTuple = read_config(
                resources.files("OSmOSE").joinpath(self.__configfile)
            )

        else:
            self.__configfile = config_file
            self.__full_config: NamedTuple = read_config(config_file)

        self.__config = self.__full_config.Job

        self.__prepared_jobs = []
        self.__ongoing_jobs = []
        self.__finished_jobs = []

        required_properties = [
            "job_scheduler",
            "env_script",
            "env_name",
            "queue",
            "nodes",
            "walltime",
            "ncpus",
            "mem",
            "outfile",
            "errfile",
        ]

        if not all(prop in self.__config._fields for prop in required_properties):
            raise ValueError(
                f"The provided configuration file is missing the following attributes: {'; '.join(list(set(required_properties).difference(set(self.__config._fields))))}"
            )

    # region Properties
    # READ-ONLY properties
    @property
    def config(self) -> dict:
        return self.__config

    @property
    def prepared_jobs(self):
        return self.__prepared_jobs

    @property
    def ongoing_jobs(self):
        self.update_job_status()
        return self.__ongoing_jobs

    @property
    def finished_jobs(self):
        self.update_job_status()
        return self.__finished_jobs

    # READ-WRITE properties
    @property
    def job_scheduler(self):
        return self.__config.job_scheduler

    @job_scheduler.setter
    def job_Scheduler(self, value: Literal["Torque", "Slurm"]):
        self.__config = self.__config._replace(job_scheduler=value)

    @property
    def env_script(self):
        return self.__config.env_script

    @env_script.setter
    def env_script(self, value):
        self.__config = self.__config._replace(env_script=value)

    @property
    def env_name(self):
        return self.__config.env_name

    @env_name.setter
    def env_name(self, value):
        self.__config = self.__config._replace(env_name=value)

    @property
    def queue(self):
        return self.__config.queue

    @queue.setter
    def queue(self, value):
        self.__config = self.__config._replace(queue=value)

    @property
    def nodes(self):
        return self.__config.nodes

    @nodes.setter
    def nodes(self, value):
        self.__config = self.__config._replace(nodes=value)

    @property
    def walltime(self):
        return self.__config.walltime

    @walltime.setter
    def walltime(self, value):
        self.__config = self.__config._replace(walltime=value)

    @property
    def ncpus(self):
        return self.__config.ncpus

    @ncpus.setter
    def ncpus(self, value):
        self.__config = self.__config._replace(ncpus=value)

    @property
    def mem(self):
        return self.__config.mem

    @mem.setter
    def mem(self, value):
        self.__config = self.__config._replace(mem=value)

    @property
    def outfile(self):
        return self.__config.outfile

    @outfile.setter
    def outfile(self, value):
        self.__config = self.__config._replace(outfile=value)

    @property
    def errfile(self):
        return self.__config.errfile

    @errfile.setter
    def errfile(self, value):
        self.__config = self.__config._replace(errfile=value)

    # endregion

    # TODO: Move to utils
    def write_configuration(self, output_file: str = None):
        """Writes the configuration to the original configuration file, or a new file if specified.

        Parameter
        ---------
            `output_file`: the path to the new configuration file (in TOML or JSON format).
        """

        output_format = (
            Path(output_file).suffix if output_file else Path(self.__configfile).suffix
        )

        match output_format:
            case "toml":
                with open(self.__configfile, "w") as outfile:
                    outfile.write(tomlkit.dumps(self.__full_config))
            case "json":
                with open(self.__configfile, "w") as outfile:
                    outfile.write(json.dumps(self.__full_config))

    def build_job_file(
        self,
        *,
        script_path: str,
        script_args: str,
        jobname: str = None,
        preset: Literal["low", "medium", "high"] = None,
        job_scheduler: Literal["Torque", "Slurm"] = None,
        env_script: str = None,
        env_name: str = None,
        queue: str = None,
        nodes: int = None,
        walltime: str = None,
        ncpus: int = None,
        mem: str = None,
        outfile: str = None,
        errfile: str = None,
        logdir: Path = None
    ) -> str:
        """Build a job file corresponding to your job scheduler.

        When a job file is created, it becomes a `prepared_job` and can be launched automatically using the `submit_job()` method
        with no arguments.

        This method can use the configuration file to set default job parameters. Use presets to quickly adjust the resource
        consumption or fill the parameters. If a parameter is left empty, it will fall back to the value in the configuration file.
        The default preset is `low` in order to save as much resource as possible.

        Parameters
        ----------
        script_path : `str`, keyword-only
            The path to the script that will be executed in the cluster job
        script_args : `str`, keyword-only
            All the arguments required by the script, as one string.
        jobname : `str`, optional, keyword-only
            The name of the job as seen on the job list (qstat, squeue, ...). The default is the script name.
        preset : `{"low", "medium", "high"}`, optional, keyword-only
            Resource preset for the job. It is best to set up the presets in the configuration file before using them,
            as the default might not correspond to your structure. The default is `low`.
        job_scheduler : `{"Torque", "Slurm"}`, optional, keyword-only
            The job scheduler to which the jobs will be submitted. As of now, only Torque (PBS) and Slurm (Sbatch) are supported.
        env_script : `str`, optional, keyword-only
            The script used to activate the conda environment. If there is no particular script, it can just be `source` or `conda activate`
        env_name : `str`, optional, keyword-only
            The name of the conda environment this job should run into.
        queue : `str`, optional, keyword-only
            The partition/queue/qol in which the job should run. It is very dependant on the cluster.
        nodes : `int`, optional, keyword-only
            The number of nodes (i.e. computers) this job will use. If it is not using a protocol like MPI or an adapted job scheduler (like PEGASUS),
            using more than one node will have no effect.
        walltime : `str`, optional, keyword-only
            The time limit of the job. If exceeded, the job will be killed by the job scheduler. It is best to view large and ask for 1.5 or 2 times
            more than the anticipated time of the job, especially for tasks with a high degree ofo variability in processing time.
        ncpus : `int`, optional, keyword-only
            The number of CPUs required per node.
        mem : `str`, optional, keyword-only
            The quantity of RAM required per CPU. Most of the time it is an integer followed by a G for Gigabytes or Mb for Megabytes.
        outfile : `str`, optional, keyword-only
            The name of the main log file which will record the standard output from the job. If only the name is provided, it will be created in the current
            directory.
        errfile : `str`, optional, keyword-only
            The name of the error log file which will record the error output from the job. If only the name is provided, it will be created in the
            current directory. If there is no error output, it will not be created.

        Returns
        -------
        job_path : `str`
            The path to the created job file.
        """
        set_umask()
        if preset and preset.lower() not in self.__config.Presets._fields:
            raise ValueError(
                f"Unrecognized preset {preset}. Valid presets are: {', '.join(self.__config.Presets._fields)}"
            )

        job_preset = getattr(self.__config.Presets, preset)

        pwd = Path(__file__).parent

        if logdir is None:
            logdir = pwd.joinpath("log_job")
            jobdir = pwd.joinpath("ongoing_jobs")
            logdir.mkdir(mode=DPDEFAULT, exist_ok=True)
            jobdir.mkdir(mode=DPDEFAULT, exist_ok=True)
        else:
            logdir.mkdir(mode=DPDEFAULT, exist_ok=True)
            jobdir = logdir
        

        job_file = ["#!/bin/bash"]

        date_id = datetime.now().strftime('%H-%M-%S')
        # region Build header
        #! HEADER
        # Getting all header variables from the parameters or the configuration defaults
        if not job_scheduler:
            job_scheduler = self.job_scheduler
        if not jobname:
            jobname = Path(script_path).name
        if not env_name:
            env_name = self.env_name

        if not outfile:
            outfile = self.outfile

        outfile = logdir.joinpath(outfile)
        if not errfile:
            errfile = self.errfile
        errfile = logdir.joinpath(errfile)

        if not queue:
            queue = self.queue if not preset else job_preset.queue
        if not nodes:
            nodes = self.nodes if not preset else job_preset.nodes
        if not walltime:
            walltime = self.walltime if not preset else job_preset.walltime
        if not ncpus:
            ncpus = self.ncpus if not preset else job_preset.ncpus
        if not mem:
            mem = self.mem if not preset else job_preset.mem

        match job_scheduler:
            case "Torque":
                prefix = "#PBS"
                name_param = "-N "
                queue_param = "-q "
                node_param = "-l nodes="
                cpu_param = "-l ncpus="
                time_param = "-l walltime="
                mem_param = "-l mem="
                outfile_param = "-o "
                errfile_param = "-e "
                outfile = str(outfile).replace("_%j", "").format(jobname)
                errfile = str(errfile).replace("_%j", "").format(jobname)
            case "Slurm":
                prefix = "#SBATCH"
                name_param = "-J "
                queue_param = "-p "
                node_param = "-n "
                cpu_param = "-c "
                time_param = "-time "
                mem_param = "-mem "
                outfile_param = "-o "
                errfile_param = "-e "
                outfile = str(outfile).format(jobname)
                errfile = str(errfile).format(jobname)
            case _:
                prefix = "#"
                name_param = ""
                queue_param = ""
                node_param = ""
                cpu_param = ""
                time_param = ""
                mem_param = ""
                outfile_param = ""
                errfile_param = ""
                warn(
                    "Unrecognized job scheduler. Please review the PBS file and or specify a job scheduler between Torque or Slurm"
                )

        job_file.append(f"{prefix} {name_param}{jobname}")
        job_file.append(f"{prefix} {queue_param}{queue}")
        job_file.append(f"{prefix} {node_param}{nodes}")
        job_file.append(f"{prefix} {cpu_param}{ncpus}")
        job_file.append(f"{prefix} {time_param}{walltime}")
        job_file.append(f"{prefix} {mem_param}{mem}")

        uid = str(
            len(glob.glob(Path(outfile).stem[:-2] + "*.out"))
            + len(self.prepared_jobs)
            + len(self.ongoing_jobs)
        ) + date_id

        outfile = Path(outfile).with_stem(f"{Path(outfile).stem}{uid}")

        job_file.append(f"{prefix} {outfile_param}{outfile}")
        errfile = Path(errfile).with_stem(f"{Path(errfile).stem}{uid}")
        job_file.append(f"{prefix} {errfile_param}{errfile}")
        # endregion

        job_file.append(f"rm -f {outfile} {errfile}")

        #! ENV
        # The env_script should contain all the command(s) needed to load the script, with the $env_name template where the environment name should be.
        job_file.append(
            Template(f"\n{env_script if env_script else self.env_script}").substitute(
                env_name=env_name
            )
        )

        #! SCRIPT
        job_file.append(f"python {script_path} {script_args}")

        #! FOOTER
        outfilename = f"{jobname}_{date_id}_{job_scheduler}_{len(os.listdir(jobdir))}.pbs"

        job_file_path = jobdir.joinpath(outfilename)

        # job_file.append(f"\nchmod 444 {outfile} {errfile}")
        job_file.append(f"\nrm {job_file_path}\n")

        #! BUILD DONE => WRITING
        with open(job_file_path, "w") as jobfile:
            jobfile.write("\n".join(job_file))

        job_info = {"path": job_file_path, "outfile": outfile, "errfile": errfile}

        self.__prepared_jobs.append(job_info)

        return job_file_path

    # TODO support multiple dependencies and job chaining (?)
    def submit_job(
        self, jobfile: str = None, dependency: str | List[str] = None
    ) -> List[str]:
        """Submits the job file to the cluster using the job scheduler written in the file name or in the configuration file.

        Parameters:
        -----------
            jobfile: `str`
                The absolute path of the job file to be submitted. If none is provided, all the prepared job files will be submitted.

            dependency: `str` or `list(str)`
                The job ID this job is dependent on. The afterok:dependency will be added to the command for all jobs submitted.

        Returns:
        --------
            A list containing the job ids of the submitted jobs.
        """

        jobinfo_list = (
            [{"path": jobfile}] if jobfile is not None else self.prepared_jobs
        )

        jobid_list = []

        if isinstance(dependency, list):
            dependency = ":".join(dependency)

        # TODO: Think about the issue when a job have too many dependencies and workaround.

        for jobinfo in jobinfo_list:
            if "torque" in str(jobinfo["path"]).lower():
                cmd = ["qsub", f"-W depend=afterok:{dependency}", str(jobinfo["path"])] if dependency else ["qsub", str(jobinfo["path"])]
                jobid = (
                    subprocess.run(
                        cmd, stdout=subprocess.PIPE
                    )
                    .stdout.decode("utf-8")
                    .rstrip("\n")
                )
                print(f'Sent command {" ".join(cmd)}')
                jobid_list.append(jobid)
            elif "slurm" in str(jobinfo["path"]).lower():
                dep = f"-d afterok:{dependency}" if dependency else ""
                jobid = (
                    subprocess.run(
                        ["sbatch", dep, jobinfo["path"]], stdout=subprocess.PIPE
                    )
                    .stdout.decode("utf-8")
                    .rstrip("\n")
                )
                jobid_list.append(jobid)

            if jobinfo in self.prepared_jobs:
                self.__prepared_jobs.remove(jobinfo)
            self.__ongoing_jobs.append(jobinfo)

        return jobid_list

    def update_job_status(self):
        """Iterates over the list of ongoing jobs and mark them as finished if the job file does not exist."""
        for jobinfo in self.__ongoing_jobs:
            if not jobinfo["path"].exists():
                self.__ongoing_jobs.remove(jobinfo)
                self.__finished_jobs.append(jobinfo)

    def update_job_access(self):
        """In case the output files are not accessible by anyone but the owner, running this once will update the permissions for anyone to read them."""

        for job_info in self.finished_jobs:
            try:
                if job_info["outfile"].exists():
                    os.chmod(job_info["outfile"], FPDEFAULT)
                if job_info["errfile"].exists():
                    os.chmod(job_info["errfile"], FPDEFAULT)
            except KeyError:
                print(f"No outfile or errfile associated with job {job_info['path']}.")

    def read_output_file(
        self, *, outtype: Literal["out", "err"] = "out", job_file_name: str = None
    ):
        """Read the content of a specific job output file, or the first in the finished list.

        Parameters
        ----------
            outtype: `{"out","err"}`, optional, keyword-only
                The type of the output file to read, whether standard (out) or error (err). The default is out.
            job_file_name: `str`, optional, keyword-only
                The path to the job_file to read, which can be retrieved easily from Job_builder.finished_jobs. If not provided, then the first element of the list will be read.
        """
        if outtype not in ["out", "err"]:
            raise ValueError(
                "The outtype must be either out for standard output file or err for error output file."
            )

        if not job_file_name:
            if len(self.finished_jobs) == 0:
                print(
                    f"There are no finished jobs in this context. Wait until the {len(self.ongoing_jobs)} ongoing jobs are done before reading the output. Otherwise, you can specify which file you wish to read."
                )
                return
            else:
                job_file_name = (
                    self.finished_jobs[0]["outfile"]
                    if outtype == "out"
                    else self.finished_jobs[0]["errfile"]
                )

        with open(job_file_name, "r") as f:
            print("".join(f.readlines()))
