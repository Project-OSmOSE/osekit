from warnings import warn
from pathlib import Path
from importlib.resources import as_file
import random
import shutil
import struct
from collections import namedtuple
from distutils.errors import UnknownFileError
import sys
from typing import Union, NamedTuple, Tuple

import json

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

import soundfile as sf
import numpy as np
from OSmOSE.config import OSMOSE_PATH


def display_folder_storage_infos(dir_path: str) -> None:
    usage = shutil.disk_usage(dir_path)
    print("Total storage space (TB):", round(usage.total / (1024**4), 1))
    print("Used storage space (TB):", round(usage.used / (1024**4), 1))
    print("-----------------------")
    print("Available storage space (TB):", round(usage.free / (1024**4), 1))


def list_not_built_datasets(datasets_folder_path: str) -> None:
    """Prints the available datasets that have not been built by the `Dataset.build()` function.

    Parameter
    ---------
    dataset_folder_path: str
        The path to the directory containing the datasets"""

    ds_folder = Path(datasets_folder_path)

    dataset_list = [
        directory for directory in ds_folder.iterdir() if ds_folder.joinpath(directory)
    ]
    list_not_built_datasets = []

    for dataset_directory in dataset_list:
        if ds_folder.joinpath(
            dataset_directory, OSMOSE_PATH.raw_audio, "original"
        ).exists():
            list_not_built_datasets.append(dataset_directory)

    print("List of the datasets not built yet:")

    for dataset in list_not_built_datasets:
        print("  - {}".format(dataset))


def read_config(raw_config: Union[str, dict, Path]) -> NamedTuple:
    """Read the given configuration file or dict and converts it to a namedtuple. Only TOML and JSON formats are accepted for now.

    Parameter
    ---------
    raw_config : `str` or `Path` or `dict`
        The path of the configuration file, or the dict object containing the configuration.

    Returns
    -------
    config : `namedtuple`
        The configuration as a `namedtuple` object.

    Raises
    ------
    FileNotFoundError
        Raised if the raw_config is a string that does not correspond to a valid path.
    TypeError
        Raised if the raw_config is anything else than a string, a PurePath or a dict.
    NotImplementedError
        Raised if the raw_config file is in YAML format
    UnknownFileError
        Raised if the raw_config file is not in TOML, JSON or YAML formats."""

    match raw_config:
        case Path():
            with as_file(raw_config) as input_config:
                raw_config = input_config

        case str():
            if not Path(raw_config).is_file:
                raise FileNotFoundError(
                    f"The configuration file {raw_config} does not exist."
                )

        case dict():
            pass
        case _:
            raise TypeError(
                "The raw_config must be either of type str, dict or Traversable."
            )

    if not isinstance(raw_config, dict):
        with open(raw_config, "rb") as input_config:
            match Path(raw_config).suffix:
                case ".toml":
                    raw_config = tomllib.load(input_config)
                case ".json":
                    raw_config = json.load(input_config)
                case ".yaml":
                    raise NotImplementedError(
                        "YAML support will eventually get there (unfortunately)"
                    )
                case _:
                    raise UnknownFileError(
                        f"The provided configuration file extension ({Path(raw_config).suffix} is not a valid extension. Please use .toml or .json files."
                    )

    return convert(raw_config)


def convert(dictionary):
    for key, value in dictionary.items():
        if isinstance(value, dict):
            dictionary[key] = convert(value)
    return namedtuple("GenericDict", dictionary.keys())(**dictionary)


def read_header(file: str) -> Tuple[int, float, int, int]:
    """Read the first bytes of a wav file and extract its characteristics.

    At the very least, only the first 44 bytes are read. If the `data` chunk is not right after the header chunk,
    the subsequent chunks will be read until the `data` chunk is found. If there is no `data` chunk, all the file will be read.

    Parameter
    ---------
    file: str
        The absolute path of the wav file whose header will be read.

    Returns
    -------
    samplerate : `int`
        The number of samples in one frame.
    frames : `float`
        The number of frames, corresponding to the file duration in seconds.
    channels : `int`
        The number of audio channels.
    sampwidth : `int`
        The sample width.

    Note
    ----
    When there is no `data` chunk, the `frames` value will fall back on the size written in the header. This can be incorrect,
    if the file has been corrupted or the writing process has been interrupted before completion.
    """
    with open(file, "rb") as fh:
        _, size, _ = struct.unpack("<4sI4s", fh.read(12))
        chunk_header = fh.read(8)
        subchunkid, _ = struct.unpack("<4sI", chunk_header)

        if subchunkid == b"fmt ":
            _, channels, samplerate, _, _, sampwidth = struct.unpack(
                "HHIIHH", fh.read(16)
            )

        chunkOffset = fh.tell()
        found_data = False
        while chunkOffset < size and not found_data:
            fh.seek(chunkOffset)
            subchunk2id, subchunk2size = struct.unpack("<4sI", fh.read(8))
            if subchunk2id == b"data":
                found_data = True

            chunkOffset = chunkOffset + subchunk2size + 8

        if not found_data:
            print(
                "No data chunk found while reading the header. Will fallback on the header size."
            )
            subchunk2size = size - 36

        sampwidth = (sampwidth + 7) // 8
        framesize = channels * sampwidth
        frames = subchunk2size / framesize

        if (size - 72) > subchunk2size:
            print(
                f"Warning : the size indicated in the header is not the same as the actual file size. This might mean that the file is truncated or otherwise corrupt.\
                \nSupposed size: {size} bytes \nActual size: {subchunk2size} bytes."
            )

        return samplerate, frames, channels, sampwidth


def safe_read(
    file_path: str, *, nan: float = 0.0, posinf: any = None, neginf: any = None
) -> Tuple[np.ndarray, int]:
    """Open a file using Soundfile and clean up the data to be used safely

    Currently, only checks for `NaN`, `inf` and `-inf` presence. The default behavior is the same as `np.nan_to_num`:
    `NaNs` are transformed into 0.0, `inf` and `-inf` are transformed into the maximum and minimum values of their dtype.

    Parameters
    ----------
        file_path: `str`
            The path to the audio file to safely read.
        nan: `float`, optional, keyword_only
            The value that will replace `NaNs`. Default is 0.0
        posinf: `any`, optional, keyword_only
            The value that will replace `inf`. Default behavior is the maximum value of the data type.
        neginf: `any`, optional, keyword_only
            The value that will replace `-inf`. Default behavior is the minimum value of the data type.

    Returns
    -------
        audio_data: `NDArray`
            The cleaned audio data as a numpy array.
        sample_rate: `int`
            The sample rate of the data."""
    audio_data, sample_rate = sf.read(file_path)

    nan_nb = sum(np.isnan(audio_data))
    if hasattr(nan_nb, "__iter__"):
        nan_nb = sum(nan_nb)  # Double sum to account for multiple channels

    if nan_nb > 0:
        warn(
            f"{nan_nb} NaN detected in file {Path(file_path).name}. They will be replaced with {nan}."
        )

    np.nan_to_num(audio_data, copy=False, nan=nan, posinf=posinf, neginf=neginf)

    return audio_data, sample_rate


def check_n_files(
    file_list: list,
    n: int,
    *,
    output_path: str = None,
    threshold_percent: float = 0.1,
    auto_normalization: bool = False,
) -> bool:
    """Check n files at random for anomalies and may normalize them.

    Currently, check if the data for wav in PCM float format are between -1.0 and 1.0. If the number of files that
    fail the test is higher than the threshold (which is 10% of n by default, with an absolute minimum of 1), all the
    dataset will be normalized and written in another file.

    Parameters
    ----------
        file_list: `list`
            The list of files to be evaluated. It must be equal or longer than n.
        n: `int`
            The number of files to evaluate. To lower resource consumption, it is advised to check only a subset of the dataset.
            10 files taken at random should provide an acceptable idea of the whole dataset.
        output_path: `str`, optional, keyword-only
            The path to the folder where the normalized files will be written. If auto_normalization is set to True, then
            it must have a value.
        threshold_percent: `float`, optional, keyword-only
            The maximum acceptable percentage of evaluated files that can contain anomalies. Understands fraction and whole numbers. Default is 0.1, or 10%
        auto_normalization: `bool`, optional, keyword_only
            Whether the normalization should proceed automatically or not if the threshold is reached. As a safeguard, the default is False.
    Returns
    -------
        normalized: `bool`
            Indicates whether or not the dataset has been normalized.
    """
    if auto_normalization and not output_path:
        raise ValueError(
            "When auto_normalization is set to True, an output path must be specified."
        )

    if threshold_percent > 1:
        threshold_percent = threshold_percent / 100

    if n > len(file_list):
        n = len(file_list)

    if "float" in str(sf.info(file_list[0])):
        threshold = max(threshold_percent * n, 1)
        bad_files = []
        for audio_file in random.sample(file_list, n):
            data, sr = safe_read(audio_file)
            if not (np.max(data) <= 1.0 and np.min(data) >= -1.0):
                bad_files.append(audio_file)

                if len(bad_files) > threshold:
                    print(
                        "The treshold has been exceeded, too many files unadequately recorded."
                    )
                    if not auto_normalization:
                        normalize = False
                        if sys.__stdin__.isatty():
                            res = input(
                                "Do you want to automatically normalize your dataset? [Y]/n"
                            )
                            if res.lower() in ["y", "yes", ""]:
                                if not output_path:
                                    output_path = input(
                                        "Please specify the path to the output folder:"
                                    )
                                normalize = True
                        if not normalize:
                            raise ValueError(
                                "You need to set auto_normalization to True to normalize your dataset automatically."
                            )

                    Path(output_path).mkdir(mode=0o770, parents=True, exist_ok=True)

                    for audio_file in file_list:
                        data, sr = safe_read(audio_file)
                        data = (
                            (data - np.mean(data)) / np.std(data)
                        ) * 0.063  # = -24dB
                        data[data > 1] = 1
                        data[data < -1] = -1

                        sf.write(
                            Path(output_path, Path(audio_file).name),
                            data=data,
                            samplerate=sr,
                        )
                        # TODO: lock in spectrum mode
                    print(
                        "All files have been normalized. Spectrograms created from them will be locked in spectrum mode."
                    )
                    return True
    return False
