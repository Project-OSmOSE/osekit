"""
Build a OSmOSE dataset
=========================

This code will format your raw data into a OSmOSE dataset
"""





# %%
# Raw data preparation
# ------------------------
# Before you can build your dataset: 
#
# - choose a dataset name (should not contain any special character, including '-'⁾ ; 
# - create the folder ``{local_working_dir}/dataset/{dataset_name}`` (or ``{local_working_dir}/dataset/{campaign_name}/{dataset_name}`` in case of a recording campaign); 
# - place in this folder your audio data, they can be individual files or contain within multiple sub-folders ; 
# - if you have any csv files (either a ``timestamp.csv`` or ``*gps*.csv`` file) should also be placed in this folder.

# %% 
# **About timestamps** 
# All timestamps from your original data (from your audio filenames or from your csv files) MUST follow the same timestamp template which should be given in ``date_template`` 

# %% 
# **About auxiliary csv files** 
# The ``*gps*.csv`` file provides the GPS track (ie latitude and longitude coordinates) of a moving hydrophone. This file must contain the term *gps* in its filename. Auxiliary csv files : they must contain headers with the following standardized names : timestamp , depth , lat , lon

# %%
# Codes
# ------------------------

# sphinx_gallery_thumbnail_path = '_static/dataset_metadata.png'


from pathlib import Path
from OSmOSE import Dataset

#####################################################
# Define dataset path and name

path_osmose_dataset = "/home6/cazaudo/Bureau/osmose_sample_datasets/"
dataset_name = "MPSU"
campaign_name = ""

#####################################################
# Define mandatory dataset metadata

date_template = "%Y%m%d_%H%M%S" # strftime format, used to build the dataset from scratch (ignore if the dataset is already built)
depth = 10
gps_coordinates = (10,10)
gps_coordinates

#####################################################
# Run the method :meth:`OSmOSE.Dataset.Dataset.build` of the class :class:`OSmOSE.Dataset.Dataset`

dataset = Dataset(dataset_path = Path(path_osmose_dataset, campaign_name, dataset_name), gps_coordinates = gps_coordinates, depth = depth, timezone='+00:00')
dataset.build(date_template = date_template , force_upload=False, number_test_bad_files=1)




