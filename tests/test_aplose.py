import os
import platform
import pandas as pd
import numpy as np

from OSmOSE.application import Aplose
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


def test_build_path(input_dataset):
    dataset = Aplose(
        dataset_path=input_dataset["main_dir"],
        dataset_sr=240,
        analysis_params=PARAMS,
        local=True,
    )
    dataset.build()
    dataset._Aplose__build_path(adjust=True, dry=True)

    print("Values of Aplose")

    print(
        "\n".join([f"{attr} : {getattr(dataset, str(attr))}" for attr in dir(dataset)])
    )

    print(dataset._get_original_after_build())

    assert dataset.path.joinpath(OSMOSE_PATH.raw_audio, "3_44100").exists()
    assert len(list(dataset.path.joinpath(OSMOSE_PATH.raw_audio, "3_44100").glob("*.wav"))) == 10
    assert dataset.audio_path == dataset.path.joinpath(OSMOSE_PATH.raw_audio, "5_240")
    assert dataset._Aplose__spectro_foldername == "adjustment_spectros"
    assert dataset.path_output_spectrogram == dataset.path.joinpath(
        OSMOSE_PATH.spectrogram, "5_240", "adjustment_spectros", "image"
    )
    assert dataset.path_output_spectrogram_matrix == dataset.path.joinpath(
        OSMOSE_PATH.spectrogram, "5_240", "adjustment_spectros", "matrix"
    )
    
    assert not dataset.path_output_spectrogram.exists()

    dataset._Aplose__build_path(adjust=False, dry=False)
    assert dataset.path_output_spectrogram == dataset.path.joinpath(
        OSMOSE_PATH.spectrogram, "5_240", "512_512_97", "image"
    )
    assert dataset.path_output_spectrogram_matrix == dataset.path.joinpath(
        OSMOSE_PATH.spectrogram, "5_240", "512_512_97", "matrix"
    )

    assert dataset.path.joinpath(OSMOSE_PATH.statistics).exists()

