"""
Explore dataset metadata
=========================

Perform summary statistics and visualize your dataset metadata
"""

from pathlib import Path
import pandas as pd

#####################################################
# Define dataset path and name

path_osmose_dataset = "/home6/cazaudo/Bureau/osmose_sample_datasets/"
dataset_name = "MPSU"
campaign_name = ""

dataset_sr = 50000
audio_duration = 30

#####################################################
# Load as dataframe your metadata

metadata_dataset = pd.read_csv(Path(path_osmose_dataset, campaign_name, dataset_name).joinpath("data","audio",f"{audio_duration}_{dataset_sr}","file_metadata.csv"))


#####################################################
# Display a summary

metadata_dataset.describe()


#####################################################
# Plot histogram of durations

metadata_dataset['duration'].hist()

