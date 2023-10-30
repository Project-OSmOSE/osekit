"""
==============================================
Tuning some parameters
==============================================

This code will show you how to tune more 
"""

from pathlib import Path
from OSmOSE import Spectrogram
import glob

#####################################################
# Define dataset path and name

path_osmose_dataset = "/home6/cazaudo/Bureau/osmose_sample_datasets/"
dataset_name = "MPSU"
campaign_name = ""

#####################################################
# Define standard parameters for spectrogram computation

spectrogram = Spectrogram(dataset_path = Path(path_osmose_dataset, campaign_name, dataset_name))

spectrogram.dataset_sr = 18000
spectrogram.spectro_duration = 15

spectrogram.nfft = 1024
spectrogram.window_size = 1024
spectrogram.overlap = 80

spectrogram.dynamic_min = -140
spectrogram.dynamic_max = -12

#####################################################
# Check size of spectrogram

spectrogram.check_spectro_size()

#####################################################
# Initialize everything needed for spectrogram computation ; in particular, this method will segment and/or resample audio files if needed

spectrogram.initialize()

wav_path = glob.glob(path_osmose_dataset+f'{dataset_name}/data/audio/{spectrogram.spectro_duration}_{spectrogram.dataset_sr}/*wav')

spectrogram.process_all_files(list_wav_to_process=wav_path)

#####################################################
# Visualize an example of spectrogram 

spectrogram_path = glob.glob(path_osmose_dataset+f'{dataset_name}/processed/spectrogram/{spectrogram.spectro_duration}_{spectrogram.dataset_sr}/{spectrogram.nfft}_{spectrogram.window_size}_{spectrogram.overlap}/image/*png')

from matplotlib import pyplot as plt
from matplotlib import image as mpimg
  
image = mpimg.imread(spectrogram_path[0])
plt.imshow(image, interpolation='nearest', aspect='auto')

ax = plt.gca()
ax.xaxis.set_tick_params(labelbottom=False)
ax.yaxis.set_tick_params(labelleft=False)
ax.set_xticks([])
ax.set_yticks([])

plt.show()

