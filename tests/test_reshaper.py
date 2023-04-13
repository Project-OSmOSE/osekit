from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta
from typing import List
from OSmOSE.cluster.audio_reshaper import *
import pytest
import csv
import os


def test_substract_timestamps():
    # Create test data
    timestamp_data = {
        "filename": ["file1", "file2", "file3", "badfile"],
        "timestamp": [
            "2022-01-01T12:00:00.000Z",
            "2022-01-01T12:01:00.000Z",
            "2022-01-01T12:02:00.000Z",
            "20220101T12:03:00.000",
        ],
    }
    input_timestamp = pd.DataFrame(data=timestamp_data)

    files = ["file1", "file2", "file3", "badfile"]

    # Test the function for the first file
    result = substract_timestamps(input_timestamp, files, 0)
    expected_result = timedelta(seconds=0)
    assert result == expected_result

    # Test the function for the second file
    result = substract_timestamps(input_timestamp, files, 1)
    expected_result = timedelta(seconds=60)
    assert result == expected_result

    # Test the function for the badly formatted  timestamp
    with pytest.raises(ValueError) as ts_error:
        substract_timestamps(input_timestamp, files, 2)

    assert (
        str(ts_error.value)
        == "time data '20220101T12:03:00.000' does not match format '%Y-%m-%dT%H:%M:%S.%fZ'"
    )


def test_reshape_errors(input_dir):
    with pytest.raises(ValueError) as e:
        reshape("/not/a/path", 15)

    assert (
        str(e.value)
        == f"The input files must either be a valid folder path or a list of file path, not {str(Path('/not/a/path'))}."
    )

    with pytest.raises(ValueError) as e:
        reshape(input_dir, 20, last_file_behavior="misbehave")

    assert (
        str(e.value)
        == "Bad value misbehave for last_file_behavior parameters. Must be one of truncate, pad or discard."
    )

    with pytest.raises(FileNotFoundError):
        reshape(input_dir, 20)  # Supposed to fail because there is no timestamp.csv


def test_reshape_smaller(input_reshape: Path, output_dir: Path):
    reshaped_list = reshape(input_files=input_reshape, chunk_size=2, output_dir_path=output_dir, write_output=False)
    assert len(reshaped_list) == 15
    # assert sf.info(reshaped_list[0]).duration == 2.0
    # assert sf.info(reshaped_list[0]).samplerate == 44100
    # assert sum(audio/samplerate for audio in reshaped_list) == 30.0

    reshape(input_files=input_reshape, chunk_size=2, output_dir_path=output_dir, write_output=True)

    reshaped_files = [output_dir.joinpath(outfile) for outfile in pd.read_csv(str(output_dir.joinpath("timestamp.csv")), header=None)[0].values]
    # reshaped_files = sorted(
    #     [x for x in Path(output_dir).iterdir() if str(x).endswith(".wav")],
    #     key=os.path.getmtime,
    # )
    assert len(reshaped_files) == 15
    assert sf.info(reshaped_files[0]).duration == 2.0
    assert sf.info(reshaped_files[0]).samplerate == 44100
    assert sum(sf.info(file).duration for file in reshaped_files) == 30.0
    assert reshaped_files[1].name == "2022-01-01T11-59-59_000.wav"

    full_input = sf.read(input_reshape.joinpath("test.wav"))[0]

    for i in range(9):
        full_input = np.concatenate(
            (full_input, sf.read(input_reshape.joinpath(f"test{i}.wav"))[0])
        )
    full_output = sf.read(reshaped_files[0])[0]
    for file in reshaped_files[1:]:
        full_output = np.concatenate((full_output, sf.read(file)[0]))

    assert np.allclose(full_input, full_output)


def test_reshape_larger(input_reshape: Path, output_dir):
    reshape(input_files=input_reshape, chunk_size=5, output_dir_path=output_dir, write_output=True)

    reshaped_files = sorted(
        [x for x in Path(output_dir).iterdir() if str(x).endswith(".wav")],
        key=os.path.getmtime,
    )
    assert len(reshaped_files) == 6
    assert sf.info(reshaped_files[0]).duration == 5.0
    assert sf.info(reshaped_files[0]).samplerate == 44100


def test_reshape_pad_last(input_reshape: Path, output_dir):
    reshape(
        input_files=input_reshape,
        chunk_size=4,
        output_dir_path=output_dir,
        last_file_behavior="pad",
        write_output=True
    )

    reshaped_files = sorted(
        [x for x in Path(output_dir).iterdir() if str(x).endswith(".wav")],
        key=os.path.getmtime,
    )
    assert len(reshaped_files) == 8
    assert sf.info(reshaped_files[0]).duration == 4.0
    assert sf.info(reshaped_files[0]).samplerate == 44100
    assert sf.info(reshaped_files[-1]).duration == 4.0


def test_reshape_truncate_last(input_reshape: Path, output_dir):
    reshape(
        input_files=input_reshape,
        chunk_size=4,
        output_dir_path=output_dir,
        last_file_behavior="truncate",
        write_output = True
    )

    reshaped_files = [output_dir.joinpath(outfile) for outfile in pd.read_csv(str(output_dir.joinpath("timestamp.csv")), header=None)[0].values]

    assert len(reshaped_files) == 8
    assert sf.info(reshaped_files[0]).duration == 4.0
    assert sf.info(reshaped_files[0]).samplerate == 44100
    assert sf.info(reshaped_files[-1]).duration == 2.0


def test_reshape_discard_last(input_reshape: Path, output_dir):
    reshape(
        input_files=input_reshape,
        chunk_size=4,
        output_dir_path=output_dir,
        last_file_behavior="discard",
        write_output=True
    )

    reshaped_files = sorted(
        [x for x in Path(output_dir).iterdir() if str(x).endswith(".wav")],
        key=os.path.getmtime,
    )
    assert len(reshaped_files) == 7
    assert sf.info(reshaped_files[0]).duration == 4.0
    assert sf.info(reshaped_files[0]).samplerate == 44100
    assert sf.info(reshaped_files[-1]).duration == 4.0


def test_reshape_offsets(input_reshape: Path, output_dir):
    reshape(
        input_files=input_reshape,
        chunk_size=6,
        output_dir_path=output_dir,
        offset_beginning=2,
        offset_end=1,
        last_file_behavior="truncate",
        verbose=True,
        write_output=True
    )

    reshaped_files = sorted(
        [x for x in Path(output_dir).iterdir() if str(x).endswith(".wav")],
        key=os.path.getmtime,
    )

    assert len(reshaped_files) == 5
    assert sf.info(reshaped_files[0]).duration == 6.0
    assert sf.info(reshaped_files[0]).samplerate == 44100

    orig_files = [
        input_reshape.joinpath(file)
        for file in os.listdir(input_reshape)
        if ".csv" not in file
    ]
    input_content_beginning = sf.read(orig_files[0])[0]
    output_content_beginning = sf.read(reshaped_files[0])[0]
    input_content_end = sf.read(orig_files[-1])[0]
    output_content_end = sf.read(reshaped_files[-1])[0]
    assert np.array_equal(
        input_content_beginning[2 * 44100 :], output_content_beginning[:44100]
    )

    assert len(input_content_end[: 2 * 44100]) == len(output_content_end[-2 * 44100 :])
    assert np.array_equal(
        input_content_end[: 2 * 44100], output_content_end[-2 * 44100 :]
    )
