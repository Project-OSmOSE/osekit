from copy import copy
import glob
import os
import json
import time
from dateutil.parser import parse
import tomlkit
import subprocess
from uuid import uuid4
from string import Template
from typing import NamedTuple, List, Literal
from warnings import warn
from pathlib import Path
from datetime import datetime, timedelta
from importlib import resources
from OSmOSE.utils import read_config, set_umask, make_path
from OSmOSE.config import *

JOB_CONFIG_TEMPLATE = namedtuple("job_config", ["job_scheduler","env_script","env_name","queue","nodes","walltime","ncpus","mem","outfile","errfile"])

class _Job_info:
    def __init__(self, job_file_path: Path, jobname: str = "", outfile:Path = None, errfile: Path = None, job_id:str = ""):
        self.path = job_file_path
        self.name = jobname
        self.id = job_id
        self.outfile = outfile
        self.errfile = errfile
    
    def __str__(self) -> str:
        info = subprocess.run(["qstat", "-fx", self.id], stdout=subprocess.PIPE).stdout.decode("utf-8")
        elapsed = info.split("resources_used.walltime = ")[1].split("\n")[0]
        return "Not implemented yet"
class Job_builder:
    def __init__(self, config_file: str = None):
        if config_file is None:
            self.__configfile = "config.toml"
            self.__full_config: dict = read_config(
                resources.files("OSmOSE").joinpath(self.__configfile)
            )

        else:
            self.__configfile = config_file
            self.__full_config: dict = read_config(config_file)

        self.__config = self.__full_config["Job"]

        self.__prepared_jobs = []
        self.__ongoing_jobs = []
        self.__finished_jobs = []
        self.__cancelled_jobs = []

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

        if not all(prop in self.__config.keys() for prop in required_properties):
            raise ValueError(
                f"The provided configuration file is missing the following attributes: {'; '.join(list(set(required_properties).difference(set(self.__config.keys()))))}"
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
    
    @property
    def cancelled_jobs(self):
        return self.__cancelled_jobs
    
    @property
    def all_jobs(self):
        return self.prepared_jobs + self.ongoing_jobs + self.finished_jobs + self.cancelled_jobs

    # READ-WRITE properties
    @property
    def job_scheduler(self):
        return self.__config["job_scheduler"]

    @job_scheduler.setter
    def job_Scheduler(self, value: Literal["Torque", "Slurm"]):
        self.__config = self.__config["job_scheduler"] = value

    @property
    def env_script(self):
        return self.__config["env_script"]

    @env_script.setter
    def env_script(self, value):
        self.__config = self.__config["env_script"] = value

    @property
    def env_name(self):
        return self.__config["env_name"]

    @env_name.setter
    def env_name(self, value):
        self.__config = self.__config["env_name"] = value

    @property
    def queue(self):
        return self.__config["queue"]

    @queue.setter
    def queue(self, value):
        self.__config = self.__config["queue"] = value

    @property
    def nodes(self):
        return self.__config["nodes"]

    @nodes.setter
    def nodes(self, value):
        self.__config = self.__config["nodes"] = value

    @property
    def walltime(self):
        return self.__config["walltime"]

    @walltime.setter
    def walltime(self, value):
        self.__config = self.__config["walltime"] = value

    @property
    def ncpus(self):
        return self.__config["ncpus"]

    @ncpus.setter
    def ncpus(self, value):
        self.__config = self.__config["ncpus"] = value

    @property
    def mem(self):
        return self.__config["mem"]

    @mem.setter
    def mem(self, value):
        self.__config = self.__config["mem"] = value

    @property
    def outfile(self):
        return self.__config["outfile"]

    @outfile.setter
    def outfile(self, value):
        self.__config = self.__config["outfile"] = value

    @property
    def errfile(self):
        return self.__config["errfile"]

    @errfile.setter
    def errfile(self, value):
        self.__config = self.__config["errfile"] = value

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
        if "Presets" in self.__config.keys():
            if preset and preset.lower() not in self.__config["Presets"].keys():
                raise ValueError(
                    f"Unrecognized preset {preset}. Valid presets are: {', '.join(self.__config['Presets'].keys())}"
                )

            job_preset = self.__config["Presets"][preset]
        else:
            preset = None

        pwd = Path(__file__).parent

        today = datetime.today().strftime("%Y_%m_%d")

        if logdir is None:
            self.log_dir = pwd.joinpath("log_job",today)
            self.jobdir = pwd.joinpath("ongoing_jobs",today)
            make_path(self.log_dir, mode=DPDEFAULT)
            make_path(self.jobdir, mode=DPDEFAULT)
        else:
            self.log_dir = logdir.joinpath(today)
            make_path(self.log_dir, mode=DPDEFAULT)
            self.jobdir = self.log_dir
        

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

        outfile = self.log_dir.joinpath(outfile)
        if not errfile:
            errfile = self.errfile
        errfile = self.log_dir.joinpath(errfile)

        if not queue:
            queue = self.queue if not preset else job_preset["queue"]
        if not nodes:
            nodes = self.nodes if not preset else job_preset["nodes"]
        if not walltime:
            walltime = self.walltime if not preset else job_preset["walltime"]
        if not ncpus:
            ncpus = self.ncpus if not preset else job_preset["ncpus"]
        if not mem:
            mem = self.mem if not preset else job_preset["mem"]

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

        uid = date_id + str(
            len(glob.glob(Path(outfile).stem[:-2] + "*.out"))
            + len(self.prepared_jobs)
            + len(self.ongoing_jobs)
        ).zfill(3)

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
        outfilename = f"{jobname}_{date_id}_{job_scheduler}_{len(os.listdir(self.jobdir))}.pbs"

        job_file_path = self.jobdir.joinpath(outfilename)

        # job_file.append(f"\nchmod 444 {outfile} {errfile}")
        job_file.append(f"\nrm {job_file_path}\n")

        #! BUILD DONE => WRITING
        with open(job_file_path, "w") as jobfile:
            jobfile.write("\n".join(job_file))

        try:
            get_dict_index_in_list(self.all_jobs, "name", jobname)
            jobname = jobname + str(len(self.all_jobs))
        except ValueError:
            pass

        self.__prepared_jobs.append(_Job_info(job_file_path, jobname, outfile, errfile))

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

        job_info_list = (
            [_Job_info(jobfile)] if jobfile is not None else copy(self.prepared_jobs)
        )

        jobid_list = []

        if isinstance(dependency, list):
            dependency = ":".join(dependency)

        # TODO: Think about the issue when a job have too many dependencies and workaround.

        for job_info in job_info_list:
            if "torque" in str(job_info.path).lower():
                cmd = ["qsub", f"-W depend=afterok:{dependency}", str(job_info.path)] if dependency else ["qsub", str(job_info.path)]
                jobid = (
                    subprocess.run(
                        cmd, stdout=subprocess.PIPE
                    )
                    .stdout.decode("utf-8")
                    .rstrip("\n")
                )
                print(f'Sent command {" ".join(cmd)}')
                jobid_list.append(jobid)

            elif "slurm" in str(job_info.path).lower():
                dep = f"-d afterok:{dependency}" if dependency else ""
                jobid = (
                    subprocess.run(
                        ["sbatch", dep, job_info.path], stdout=subprocess.PIPE
                    )
                    .stdout.decode("utf-8")
                    .rstrip("\n")
                )
                jobid_list.append(jobid)

            if job_info in self.prepared_jobs:
                self.__prepared_jobs.remove(job_info)
            job_info.id = jobid
            self.__ongoing_jobs.append(job_info)

        return jobid_list

    def update_job_status(self):
        """Iterates over the list of ongoing jobs and mark them as finished if the job file does not exist."""
        for job_info in self.__ongoing_jobs:
            if not job_info.path.exists():
                self.__ongoing_jobs.remove(job_info)
                self.__finished_jobs.append(job_info)

    def update_job_access(self):
        """In case the output files are not accessible by anyone but the owner, running this once will update the permissions for anyone to read them."""

        for job_info in self.finished_jobs:
            try:
                if job_info.outfile.exists():
                    os.chmod(job_info.outfile, FPDEFAULT)
                if job_info.errfile.exists():
                    os.chmod(job_info.errfile, FPDEFAULT)
            except KeyError:
                print(f"No outfile or errfile associated with job {job_info.path}.")

    def list_jobs(self):
        res = ""
        epoch = datetime.utcfromtimestamp(0)
        today = datetime.strftime(datetime.today(), "%d%m%y")
        if len(self.prepared_jobs) > 0:
            res += "==== PREPARED JOBS ====\n\n"
        for job_info in self.prepared_jobs:
            created_at = datetime.strptime(today + job_info.outfile.stem[-11:-3], "%d%m%y%H-%M-%S")
            res += f"{job_info.name} (created at {created_at}) : ready to start.\n"

        if len(self.ongoing_jobs) > 0:
            res += "==== ONGOING JOBS ====\n\n"
        for job_info in self.ongoing_jobs:
            created_at = datetime.strptime(today + job_info.outfile.stem[-11:-3], "%d%m%y%H-%M-%S")
            delta = datetime.now() - created_at
            strftime = f"{'%H hours, ' if delta.seconds >= 3600 else ''}{'%M minutes and ' if delta.seconds >= 60 else ''}%S seconds"
            elapsed_time = datetime.strftime(epoch + delta, strftime)
            res += f"{job_info.name} (created at {created_at}) : running for {elapsed_time}.\n"

        if len(self.finished_jobs) > 0:
            res += "==== FINISHED JOBS ====\n\n"
        for job_info in self.finished_jobs:
            created_at = datetime.strptime(today + job_info.outfile.stem[-11:-3], "%d%m%y%H-%M-%S")

            if not job_info.outfile.exists():
                res += f"{job_info.name} (created at {created_at}) : Output file still writing..."
            else:
                delta = datetime.fromtimestamp(time.mktime(time.localtime(job_info.outfile.stat().st_ctime))) - created_at
                strftime = f"{'%H hours, ' if delta.seconds >= 3600 else ''}{'%M minutes and ' if delta.seconds >= 60 else ''}%S seconds"
                elapsed_time = datetime.strftime(epoch + delta, strftime)
                res += f"{job_info.name} (created at {created_at}) : finished in {elapsed_time}.\n"

        if len(self.cancelled_jobs) > 0:
            res += "==== CANCELLED JOBS ====\n\n"
            for job_info in self.cancelled_jobs:
                res+= f"{job_info.name}"
        print(res)

    def search_jobs(self, *, date:str, log_dir: Path= None):
        top_dir = log_dir if log_dir is not None else self.log_dir
        if top_dir is None:
            raise ValueError("If no job has been submitted by this instance of Job_builder, then log_dir must be specified.")
        
        date = parse(date, fuzzy=True)
        foldername = datetime.stftime(date, "%Y_%m_%d")

        return [output for output in os.listdir(top_dir.joinpath(foldername)) if output.endswith(".out")]

    def display_job_info(self, *, job_identifier:str):
        print(self.get_ongoing_job(job_identifier=job_identifier))


    def read_output_file(
        self, *, job_name: str = None, outtype: Literal["out", "err"] = "out", job_file_name: str = None
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
            job_id = 0
            if job_name:
                job_info = get_dict_index_in_list(self.finished_jobs, "name", job_name.rstrip())
            elif len(self.finished_jobs) == 0:
                print(
                    f"There are no finished jobs in this context. Wait until the {len(self.ongoing_jobs)} ongoing jobs are done before reading the output. Otherwise, you can specify which file you wish to read."
                )
                return
            
            job_file_name = (
                job_info.outfile
                if outtype == "out"
                else job_info.errfile
            )
        
        if not Path(job_file_name).exists():
            print("The output file has not been written yet. Please wait a few moments.")
            return

        with open(job_file_name, "r") as f:
            print("".join(f.readlines()))

    def clear(self, all_jobs:bool=False):
        """Clear the queue of any job stuck.
        
        Parameter
        ---------
            all: bool, optional, keyword-only
                If set to True, will clear the job history of the job builder."""
        if all_jobs:
            for job_info in self.all_jobs:
                if job_info.path.exists():
                    job_info.path.unlink()

        for job_info in self.ongoing_jobs:
            res = subprocess.run(f"qstat {job_info.id}", shell=True, stdout=subprocess.PIPE).stdout.decode("utf-8").rstrip("\n")
            if f"{job_info.id} has finished" in res and job_info.path.exists():
                job_info.path.unlink()
                
    def delete_job(self, job_identifier:str):
        job_info = self.get_ongoing_job(job_identifier=job_identifier)
        
        if "Torque" in str(job_info.path):
            subprocess.run(["qdel", job_info.id])
        elif "Slurm" in str(job_info.path):
            subprocess.run(["scancel", job_info.id])
        self.__ongoing_jobs.remove(job_info)
        self.__cancelled_jobs.append(job_info)

    def get_ongoing_job(self, job_identifier:str):
        try:
            return get_dict_index_in_list(self.ongoing_jobs, "name", job_identifier.rstrip())
        except ValueError:
            return get_dict_index_in_list(self.ongoing_jobs, "id", job_identifier.rstrip())


def get_dict_index_in_list(item_list: list, attr: str, value: any) -> int:
    for item in item_list:
        if getattr(item, attr) == value:
            return item
    
    raise ValueError(f"The value {value} appears nowhere in the {attr} list.")