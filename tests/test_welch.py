import os
import platform
import pandas as pd
import numpy as np
import pytest

from OSmOSE.features import Welch
from OSmOSE.config import OSMOSE_PATH
import soundfile as sf

PARAMS = {
    "nfft": 512,
    "window_size": 512,
    "overlap": 97,
    "colormap": "viridis",
    "zoom_level": 2,
    "number_adjustment_spectrogram": 2,
    "dynamic_min": 0,
    "dynamic_max": 150,
    "spectro_duration": 5,
    "data_normalization": "instrument",
    "HPfilter_min_freq": 0,
    "sensitivity_dB": -164,
    "peak_voltage": 2.5,
    "spectro_normalization": "density",
    "gain_dB": 14.7,
    "zscore_duration": "original",
}


def test_init(input_dataset):
    # no sr
    with pytest.raises(ValueError) as e:
        Welch(dataset_path=input_dataset)
    assert str(e.value) == "If you dont know your sr, please use the build() method first"
