import os
import platform
import pandas as pd
import numpy as np

from OSmOSE.application import Aplose
from OSmOSE.config import OSMOSE_PATH
from pathlib import Path
import soundfile as sf
import pytest

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

@pytest.mark.unit
def test_build_path(input_dataset):
    dataset = Aplose(
        dataset_path=input_dataset["main_dir"],
        dataset_sr=240,
        analysis_params=PARAMS,
        local=True,
    )
    dataset.build()
    dataset.build_path(adjust=True, dry=True)

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

    dataset.build_path(adjust=False, dry=False)
    assert dataset.path_output_spectrogram == dataset.path.joinpath(
        OSMOSE_PATH.spectrogram, "5_240", "512_512_97", "image"
    )
    assert dataset.path_output_spectrogram_matrix == dataset.path.joinpath(
        OSMOSE_PATH.spectrogram, "5_240", "512_512_97", "matrix"
    )

    assert dataset.path.joinpath(OSMOSE_PATH.statistics).exists()

@pytest.mark.integ
def test_initialize_5s(input_dataset):
    sr = 44100
    dataset = Aplose(
        dataset_path=input_dataset["main_dir"],
        dataset_sr=sr,
        analysis_params=PARAMS,
        local=True,
    )

    dataset.initialize()

    timestamp_path = dataset.path.joinpath(
        OSMOSE_PATH.raw_audio.joinpath(f"5_{sr}", "timestamp.csv")
    )

    spectro_paths = [
        OSMOSE_PATH.spectrogram.joinpath(f"5_{sr}", "512_512_97", "image"),
        OSMOSE_PATH.spectrogram.joinpath(f"5_{sr}", "512_512_97", "matrix"),
        OSMOSE_PATH.spectrogram.joinpath("adjust_metadata.csv"),
        OSMOSE_PATH.raw_audio.joinpath(f"5_{sr}"),
        OSMOSE_PATH.raw_audio.joinpath(f"5_{sr}", "metadata.csv"),
        timestamp_path,
    ]

    for path in spectro_paths:
        assert dataset.path.joinpath(path).resolve().exists()

@pytest.mark.integ
def test_initialize_2s(input_dataset):
    PARAMS["spectro_duration"] = 2
    sr = 44100
    dataset = Aplose(
        dataset_path=input_dataset["main_dir"],
        dataset_sr=sr,
        analysis_params=PARAMS,
        local=True,
    )

    dataset.initialize()

    timestamp_path = dataset.path.joinpath(
        OSMOSE_PATH.raw_audio.joinpath(f"2_{sr}", "timestamp.csv")
    )

    spectro_paths = [
        OSMOSE_PATH.spectrogram.joinpath(f"2_{sr}", "512_512_97", "image"),
        OSMOSE_PATH.spectrogram.joinpath(f"2_{sr}", "512_512_97", "matrix"),
        OSMOSE_PATH.spectrogram.joinpath("adjust_metadata.csv"),
        OSMOSE_PATH.raw_audio.joinpath(f"2_{sr}"),
        OSMOSE_PATH.raw_audio.joinpath(f"2_{sr}", "metadata.csv"),
        timestamp_path,
    ]

    for path in spectro_paths:
        assert dataset.path.joinpath(path).resolve().exists()

@pytest.mark.integ
def test_generate_spectrogram(input_dataset):
    dataset = Aplose(
        dataset_path=input_dataset["main_dir"],
        dataset_sr=44100,
        analysis_params=PARAMS,
        local=True,
    )

    dataset.zoom_level = 0
    dataset.spectro_duration = 3

    dataset.initialize()

    file_to_process = Path(next(dataset.path_input_audio_file.glob("*.wav")))
    
    with pytest.raises(ValueError) as e:
        dataset.generate_spectrogram(audio_file=file_to_process)
    assert str(e.value) == "Neither image or matrix are set to be generated. Please set at least one of save_matrix or save_image to True to proceed with the spectrogram generation, or use the welch() method to get the raw data."

    dataset.generate_spectrogram(audio_file=file_to_process, save_image=True)

    result = os.listdir(dataset.path_output_spectrogram)
    assert len(result) == 1
    assert result[0] == f"{file_to_process.name}_1_0.png"

@pytest.mark.reg
def test_spectro_creation(output_dir):
    pass

