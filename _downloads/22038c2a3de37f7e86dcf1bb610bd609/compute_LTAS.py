"""
From spectrograms to LTAS (Long Term Averaged Spectrograms)
============================================================

This code will show you how to compute LTAS from audio file-level spectrograms
"""




# %%
# Prerequisites
# ------------------------
# You first need to compute audio file-level spectrograms before computing LTAS ; see the use case :ref:`sphx_glr_gallery_tutorials_Spectrogram_generate_spectrogram.py` to do this.

# %%
# Codes
# ------------------------




from pathlib import Path
from OSmOSE import Spectrogram
import glob

#####################################################
# Define dataset path and name

path_osmose_dataset = "/home6/cazaudo/Bureau/osmose_sample_datasets/"
dataset_name = "MPSU"
campaign_name = ""

#####################################################
# Define standard parameters for LTAS computation

spectrogram = Spectrogram(dataset_path = Path(path_osmose_dataset, campaign_name, dataset_name))

dataset_sr = 50000
time_res = 30
time_scale = 'all'

spectrogram.build_LTAS(time_resolution=time_res , sample_rate = dataset_sr, time_scale=time_scale)


#####################################################
# Visualize an example of LTAS 

spectrogram_path = glob.glob(path_osmose_dataset+f'{dataset_name}/processed/LTAS/LTAS_{time_scale}.png')

from matplotlib import pyplot as plt
from matplotlib import image as mpimg
  
image = mpimg.imread(spectrogram_path[0])
plt.imshow(image)

ax = plt.gca()
ax.xaxis.set_tick_params(labelbottom=False)
ax.yaxis.set_tick_params(labelleft=False)
ax.set_xticks([])
ax.set_yticks([])

plt.show()


