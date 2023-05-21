import os
from pathlib import Path

import pytest
from OSmOSE import Job_builder
from OSmOSE.utils import read_config, convert

custom_config = convert({
    "job_scheduler":"Torque",
    "env_script":"conda activate",
    "env_name":"osmose",
    "queue":"normal",
    "nodes":2,
    "walltime":"12:00:00",
    "ncpus":16,
    "mem":"42G",
    "outfile":"%j.out",
    "errfile":"%j.err"
})

pbshead = """#!/bin/bash
#PBS -N test_job
#PBS -q normal
#PBS -l nodes=2
#PBS -l ncpus=16
#PBS -l walltime=12:00:00
#PBS -l mem=42G"""

def test_build_job_file(output_dir):
    script_path = "/path/to/script"
    script_args = "--arg1 value1 --arg2 value2"
    jobname = "test_job"
    
    jb = Job_builder()
    jb._Job_builder__config = custom_config
    jb.build_job_file(script_path = script_path, script_args=script_args, jobname=jobname, logdir=output_dir)

    with open(jb.prepared_jobs[0]["path"], "r") as f:
        text = "".join(f.readlines())
    
    assert pbshead in text