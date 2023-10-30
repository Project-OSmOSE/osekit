"""
====================================================
Workflow for wind speed estimation using ERA5 data
====================================================

This code builds a workflow to build a parametric model for wind speed estimation based on welch spectra and ERA5 wind speed data
"""


# %%
# Prerequisites
# ------------------------
# You need to have in your OSmOSE dataset the joined dataframe containing welch spectra paths and ERA5 data, saved as a csv file located in `<path_osmose_dataset>/<dataset_name>/processed/auxiliary/aux_data.csv` ; see the use case :ref:`sphx_glr_gallery_tutorials_Auxiliary_joiner_auxiliary.py` to do this.



# %%
# Codes
# ------------------------

#####################################################
# Define dataset path and name

from pathlib import Path
from OSmOSE.Weather import Weather


path_osmose_dataset = "/home6/cazaudo/Bureau/osmose_sample_datasets/"
dataset_name = "SES1"
campaign_name = ""

date_template = "%Y%m%d_%H%M%S" 

#####################################################
# Select your set of welch spectra through their time resolution and sampling rate

time_resolution_welch = 300
sample_rate_welch = 38400

#####################################################
# Run the Weather class to do the workflow

appli_weather = Weather(path_osmose_dataset,dataset_name,time_resolution_welch,sample_rate_welch)

appli_weather.save_all_welch()

appli_weather.append_SPL_filtered(freq_min=7500,freq_max=8500)

appli_weather.wind_speed_estimation()


#####################################################
# Visualize an example of results 

temporal_ecmwf_model = path_osmose_dataset+f'{dataset_name}/appli/weather/temporal_ecmwf_model.png'

from matplotlib import pyplot as plt
from matplotlib import image as mpimg

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
  
image = mpimg.imread(temporal_ecmwf_model)
plt.imshow(image, interpolation='nearest', aspect='auto')

ax = plt.gca()
ax.xaxis.set_tick_params(labelbottom=False)
ax.yaxis.set_tick_params(labelleft=False)
ax.set_xticks([])
ax.set_yticks([])

plt.show()



