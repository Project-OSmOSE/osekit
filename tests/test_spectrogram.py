import os
import platform
import pandas as pd
import numpy as np

from OSmOSE import Spectrogram
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
    dataset = Spectrogram(
        dataset_path=input_dataset["main_dir"],
        dataset_sr=240,
        analysis_params=PARAMS,
        local=True,
    )
    dataset.build()
    dataset._Spectrogram__build_path(adjust=True, dry=True)

    print("Values of spectrogram")

    print(
        "\n".join([f"{attr} : {getattr(dataset, str(attr))}" for attr in dir(dataset)])
    )

    print(dataset._get_original_after_build())

    assert dataset.path.joinpath(OSMOSE_PATH.raw_audio, "3_44100").exists()
    assert len(list(dataset.path.joinpath(OSMOSE_PATH.raw_audio, "3_44100").glob("*.wav"))) == 10
    assert dataset.audio_path == dataset.path.joinpath(OSMOSE_PATH.raw_audio, "5_240")
    assert dataset._Spectrogram__spectro_foldername == "adjustment_spectros"
    assert dataset.path_output_spectrogram == dataset.path.joinpath(
        OSMOSE_PATH.spectrogram, "5_240", "adjustment_spectros", "image"
    )
    assert dataset.path_output_spectrogram_matrix == dataset.path.joinpath(
        OSMOSE_PATH.spectrogram, "5_240", "adjustment_spectros", "matrix"
    )
    
    assert not dataset.path_output_spectrogram.exists()

    dataset._Spectrogram__build_path(adjust=False, dry=False)
    assert dataset.path_output_spectrogram == dataset.path.joinpath(
        OSMOSE_PATH.spectrogram, "5_240", "512_512_97", "image"
    )
    assert dataset.path_output_spectrogram_matrix == dataset.path.joinpath(
        OSMOSE_PATH.spectrogram, "5_240", "512_512_97", "matrix"
    )

    assert dataset.path.joinpath(OSMOSE_PATH.statistics).exists()


def test_initialize_5s(input_dataset):
    sr = 44100 if platform.system() else 240
    dataset = Spectrogram(
        dataset_path=input_dataset["main_dir"],
        dataset_sr=sr,
        analysis_params=PARAMS,
        local=True,
    )

    dataset.initialize(reshape_method="classic")

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

    print(os.listdir(dataset.path.joinpath(OSMOSE_PATH.raw_audio, f"5_{sr}")))

    for path in spectro_paths:
        assert dataset.path.joinpath(path).resolve().exists()

    all_audio_files = list(dataset.audio_path.glob("*.wav"))

    assert len(all_audio_files) == 6
    for file in all_audio_files:
        assert sf.info(file).samplerate == sr
        assert sf.info(file).duration == 5.0

    full_input = np.concatenate(
        (
            tuple(
                [
                    sf.read(
                        dataset.path.joinpath(
                            OSMOSE_PATH.raw_audio, "3_44100", f"test_{i}.wav"
                        )
                    )[0]
                    for i in range(10)
                ]
            )
        )
    )

    csvFileArray = pd.read_csv(timestamp_path, header=None)

    filename_csv = csvFileArray[0].values

    full_output = np.concatenate(
        tuple([sf.read(dataset.audio_path.joinpath(file))[0] for file in filename_csv])
    )

    assert np.allclose(full_input, full_output)


def test_initialize_2s(input_dataset):
    PARAMS["spectro_duration"] = 2
    sr = 44100 if platform.system() else 240
    dataset = Spectrogram(
        dataset_path=input_dataset["main_dir"],
        dataset_sr=sr,
        analysis_params=PARAMS,
        local=True,
    )

    dataset.initialize(reshape_method="classic")

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

    all_audio_files = list(dataset.audio_path.glob("*wav"))

    assert len(all_audio_files) == 15
    for file in all_audio_files:
        assert sf.info(file).samplerate == sr
        assert sf.info(file).duration == 2.0

    full_input = np.concatenate(
        (
            tuple(
                [
                    sf.read(
                        dataset.path.joinpath(
                            OSMOSE_PATH.raw_audio, "3_44100", f"test_{i}.wav"
                        )
                    )[0]
                    for i in range(10)
                ]
            )
        )
    )

    csvFileArray = pd.read_csv(timestamp_path, header=None)

    filename_csv = csvFileArray[0].values

    full_output = np.concatenate(
        tuple([sf.read(dataset.audio_path.joinpath(file))[0] for file in filename_csv])
    )

    assert np.allclose(full_input, full_output)