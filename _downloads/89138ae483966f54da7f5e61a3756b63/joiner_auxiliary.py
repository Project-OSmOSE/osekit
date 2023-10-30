

"""
==============================================
Join welch spectra with ERA5 data
==============================================

This code will join welch spectra with variables from ERA5 within a pandas dataframe
"""


# %%
# ERA5 downloading
# ------------------------
# To use this code you will first need to download and format some ERA5 data : to do so, please use this `notebook <./download_ERA5.html>`__ . For OSmOSE members, this notebook can be directly executing
# on our `Google drive team  <https://drive.google.com/drive/folders/1QtNjUo1EaGEKSs4BY_E9iRUSWAlw4bOs>`_ 


# %%
# Codes
# ------------------------

#####################################################
# Define dataset path and name

from pathlib import Path
from OSmOSE.Auxiliary import Auxiliary


path_osmose_dataset = "/home6/cazaudo/Bureau/osmose_sample_datasets/"
dataset_name = "SES1"
campaign_name = ""

date_template = "%Y%m%d_%H%M%S" 

#####################################################
# Select your set of welch spectra through their time resolution and sampling rate

time_resolution_welch = 300
sample_rate_welch = 38400

#####################################################
# Run the Auxiliary class to perform joining

joiner = Auxiliary(path_osmose_dataset,dataset_name,time_resolution_welch,sample_rate_welch)

joiner.join_welch()

joiner.join_era()

joiner.save_aux_data()


