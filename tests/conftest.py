from pathlib import Path
import pytest
import numpy as np
import soundfile as sf
import shutil
import csv
from OSmOSE.config import OSMOSE_PATH

def capture_csv(monkeypatch):
    pass


@pytest.fixture
def input_dataset(tmp_path: Path):
    main_dir = tmp_path.joinpath("sample_dataset")
    main_audio_dir = main_dir.joinpath(OSMOSE_PATH.raw_audio)
    orig_audio_dir = main_audio_dir.joinpath("original")
    process_dir = main_dir.joinpath(OSMOSE_PATH.spectrogram)

    folders_to_create = [main_dir, main_audio_dir, orig_audio_dir, process_dir]

    for folder in folders_to_create:
        folder.mkdir(parents=True)

    rate = 44100  # samples per second
    duration = 3
    rng = np.random.default_rng()

    for i in range(10):
        data = rng.standard_normal(duration * rate)
        data[data > 1] = 1
        data[data < -1] = -1
        wav_file = orig_audio_dir.joinpath(f"test-{i}.wav")
        sf.write(wav_file, data, rate, format="WAV", subtype="DOUBLE")

        with open(
            orig_audio_dir.joinpath("timestamp.csv"), "a", newline=""
        ) as timestampf:
            writer = csv.writer(timestampf)
            writer.writerow(
                [
                    str(wav_file),
                    f"2022-01-01T12:00:{str(3*i).zfill(2)}.000Z",
                    "UTC",
                ]
            )

    yield dict(
        zip(
            ["main_dir", "main_audio_dir", "orig_audio_dir", "process_dir"],
            folders_to_create,
        )
    )


@pytest.fixture
def input_dir(tmp_path):
    # Parameters
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    rate = 44100  # samples per second
    duration = 3
    rng = np.random.default_rng()

    data = rng.standard_normal(duration * rate)
    data[data > 1] = 1
    data[data < -1] = -1
    wav_file = input_dir / "test.wav"
    sf.write(wav_file, data, rate, format="WAV", subtype="DOUBLE")

    yield input_dir


@pytest.fixture
def output_dir(tmp_path: Path):
    output_dir = tmp_path.joinpath("output")
    if output_dir.is_dir():
        shutil.rmtree(output_dir)
    output_dir.mkdir()
    yield output_dir


@pytest.fixture
def input_spectrogram(input_dataset):
    analysis_params = {
        "nfft": 512,
        "winsize": 512,
        "overlap": 97,
        "spectro_colormap": "viridis",
        "zoom_levels": 2,
        "number_adjustment_spectrograms": 2,
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

    yield input_dataset, analysis_params


@pytest.fixture
def input_reshape(input_dir: Path):
    for i in range(9):
        wav_file = input_dir.joinpath(f"test{i}.wav")
        shutil.copyfile(input_dir.joinpath("test.wav"), wav_file)

    with open(input_dir.joinpath("timestamp.csv"), "w", newline="") as timestampf:
        writer = csv.writer(timestampf)
        writer.writerow(
            [str(input_dir.joinpath("test.wav")), "2022-01-01T11:59:57.000Z", "UTC"]
        )
        writer.writerows(
            [
                [
                    str(input_dir.joinpath(f"test{i}.wav")),
                    f"2022-01-01T12:00:{str(3*i).zfill(2)}.000Z",
                    "UTC",
                ]
                for i in range(9)
            ]
        )

    return input_dir
