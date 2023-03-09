import sys
import shutil
from datetime import datetime, timedelta
from typing import List, Union, Literal
from argparse import ArgumentParser
from pathlib import Path

import soundfile as sf
import numpy as np
import pandas as pd


def substract_timestamps(
    input_timestamp: pd.DataFrame, files: List[str], index: int
) -> timedelta:
    """Substracts two timestamp_list from the "timestamp" column of a dataframe at the indexes of files[i] and files[i-1] and returns the time delta between them

    Parameters:
    -----------
        input_timestamp: the pandas DataFrame containing at least two columns: filename and timestamp

        files: the list of file names corresponding to the filename column of the dataframe

        index: the index of the file whose timestamp will be substracted

    Returns:
    --------
        The time between the two timestamp_list as a datetime.timedelta object"""

    if index == 0:
        return timedelta(seconds=0)

    cur_timestamp: str = input_timestamp[input_timestamp["filename"] == files[index]][
        "timestamp"
    ].values[0]
    cur_timestamp: datetime = datetime.strptime(cur_timestamp, "%Y-%m-%dT%H:%M:%S.%fZ")
    next_timestamp: str = input_timestamp[
        input_timestamp["filename"] == files[index + 1]
    ]["timestamp"].values[0]
    next_timestamp: datetime = datetime.strptime(
        next_timestamp, "%Y-%m-%dT%H:%M:%S.%fZ"
    )

    return next_timestamp - cur_timestamp


# TODO: what to do with overlapping last file ? Del or incomplete ?
def reshape(
    input_files: Union[str, list],
    chunk_size: int,
    *,
    output_dir_path: str = None,
    batch_ind_min: int = 0,
    batch_ind_max: int = -1,
    max_delta_interval: int = 5,
    last_file_behavior: Literal["truncate", "pad", "discard"] = "pad",
    offset_beginning: int = 0,
    offset_end: int = 0,
    verbose: bool = False,
    overwrite: bool = False,
    force_reshape: bool = False,
) -> List[str]:
    """Reshape all audio files in the folder to be of the specified duration. If chunk_size is superior to the base duration of the files, they will be fused according to their order in the timestamp.csv file in the same folder.

    Parameters:
    -----------
        input_files: `str` or `list(str)`
            Either the directory containing the audio files and the timestamp.csv file, in which case all audio files will be considered,
            OR a list of audio files all located in the same directory alongside a timestamp.csv, in which case only they will be used.

        chunk_size: `int`
            The target duration for all the reshaped files, in seconds.

        output_dir_path: `str`, optional, keyword-only
            The directory where the newly created audio files will be created. If none is provided,
            it will be the same as the input directory. This is not recommended.

        batch_ind_min: `int`, optional, keyword-only
            The first file of the list to be processed. Default is 0.

        batch_ind_max: `int`, optional, keyword-only
            The last file of the list to be processed. Default is -1, meaning the entire list is processed.

        max_delta_interval: `int`, optional, keyword-only
            The maximum number of second allowed for a delta between two timestamp_list to still be considered the same.
            Default is 5s up and down.

        last_file_behavior: `{"truncate","pad","discard"}, optional, keyword-only
            Tells the reshaper what to do with if the last data of the last file is too small to fill a whole file.
            This parameter is only active if `batch_ind_max` is `-1`
            - `truncate` creates a truncated file with the remaining data, which will have a different duration than the others.
            - `pad` creates a file of the same duration than the others, where the missing data is filled with 0.
            - `discard` ignores the remaining data. The last seconds/minutes/hours of audio will be lost in the reshaping.
        The default methodd is `pad`.

        offset_beginning: `int`, optional, keyword-only
            The number of seconds that should be skipped in the first input file. When parallelising the reshaping,
            it would mean that the beginning of the file is being processed by another job. Default is 0.

        offset_end: `int`, optional, keyword-only
            The number of seconds that should be ignored in the last input file. When parallelising the reshaping, it would mean that the end of this file is processed by another job.
            Default is 0, meaning that nothing is ignored.

        verbose: `bool`, optional, keyword-only
            Whether to display informative messages or not.

        overwrite: `bool`, optional, keyword-only
            Deletes the content of `output_dir_path` before writing the results. If it is implicitly the `input_files` directory,
            nothing happens. WARNING: If `output_dir_path` is explicitly set to be the same as `input_files`, then it will be overwritten!

        force_reshape: `bool`, optional, keyword-only
            Ignore all warnings and non-fatal errors while reshaping.
    Returns:
    --------
        The list of the path of newly created audio files.
    """
    files = []

    #! Validation
    if last_file_behavior not in ["truncate", "pad", "discard"]:
        raise ValueError(
            f"Bad value {last_file_behavior} for last_file_behavior parameters. Must be one of truncate, pad or discard."
        )

    if isinstance(input_files, list):
        input_dir_path = Path(input_files[0]).parent
        files = [Path(file).stem for file in input_files]
        if verbose:
            print(f"Input directory detected as {input_dir_path}")
    else:
        input_dir_path = Path(input_files)

    if not input_dir_path.is_dir():
        raise ValueError(
            f"The input files must either be a valid folder path or a list of file path, not {str(input_dir_path)}."
        )

    if not input_dir_path.joinpath("timestamp.csv").exists():
        raise FileNotFoundError(
            f"The timestamp.csv file must be present in the directory {input_dir_path} and correspond to the audio files in the same location."
        )

    if overwrite and output_dir_path:
        shutil.rmtree(output_dir_path)

    if not output_dir_path:
        print("No output directory provided. Will use the input directory instead.")
        output_dir_path = input_dir_path
        if overwrite:
            print(
                "Cannot overwrite input directory when the output directory is implicit! Choose a different output directory instead."
            )
    else:
        output_dir_path = Path(output_dir_path)

    output_dir_path.mkdir(mode=770, parents=True, exists_ok=True)

    input_timestamp = pd.read_csv(
        input_dir_path.joinpath("timestamp.csv"),
        header=None,
        names=["filename", "timestamp", "timezone"],
    )

    # When automatically reshaping, will populate the files list
    if not files:
        files = list(
            input_timestamp["filename"][
                batch_ind_min : batch_ind_max
                if batch_ind_max > 0
                else input_timestamp.size
            ]
        )

    if verbose:
        print(f"Files to be reshaped: {','.join(files)}")

    result = []
    timestamp_list = []
    timestamp: datetime = None
    previous_audio_data = np.empty(1)
    sample_rate = 0
    i = 0
    t = 0
    proceed = force_reshape  # Default is False

    while i < len(files):
        audio_data, sample_rate = sf.read(input_dir_path.joinpath(files[i]))
        if i == 0:
            timestamp = input_timestamp[input_timestamp["filename"] == files[i]][
                "timestamp"
            ].values[0]
            timestamp = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S.%fZ")
            audio_data = audio_data[offset_beginning * sample_rate :]
        elif i == len(files) - 1 and offset_end != 0:
            audio_data = audio_data[: len(audio_data) - (offset_end * sample_rate)]

        # Need to check if size > 1 because numpy arrays are never empty urgh
        if previous_audio_data.size > 1:
            audio_data = np.concatenate((previous_audio_data, audio_data))
            previous_audio_data = np.empty(1)

        #! AUDIO DURATION > CHUNK SIZE
        # While the duration of the audio is longer than the target chunk, we segment it into small files
        # This means to account for the creation of 10s long files from big one and not overload audio_data.
        if len(audio_data) > chunk_size * sample_rate:
            while len(audio_data) > chunk_size * sample_rate:
                output = audio_data[: chunk_size * sample_rate]
                previous_audio_data = audio_data[chunk_size * sample_rate :]

                end_time = (
                    (t + 1) * chunk_size
                    if chunk_size * sample_rate <= len(output)
                    else t * chunk_size + len(output) // sample_rate
                )

                outfilename = (
                    output_dir_path.joinpath(
                        f"reshaped_from_{t * chunk_size}_to_{end_time}_sec.wav"
                    ),
                )

                result.append(outfilename.stem)

                timestamp_list.append(
                    datetime.strftime(timestamp, "%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
                )
                timestamp += timedelta(seconds=chunk_size)

                sf.write(outfilename, output, sample_rate)

                if verbose:
                    print(
                        f"{outfilename} written! File is {(len(output)/sample_rate)} seconds long. {(len(previous_audio_data)/sample_rate)} seconds left from slicing."
                    )

                t += 1
                audio_data = previous_audio_data

            # If after we get out of the previous while loop we don't have any audio_data left, then we look at the next file.
            if len(audio_data) == 0:
                i += 1
                continue
        #! AUDIO DURATION == CHUNK SIZE
        # Else if audio_data is already in the desired duration, output it
        if len(audio_data) == chunk_size * sample_rate:
            output = audio_data
            previous_audio_data = np.empty(1)

        #! AUDIO DURATION < CHUNK_SIZE
        # Else it is shorter, then while the duration is shorter than the desired chunk,
        # we read the next file and append it to the current one.
        elif len(audio_data) < chunk_size * sample_rate:
            # If it is the last file but the audio_data is shorter than the desired chunk, then fill the remaining space with silence.
            if i == len(files) - 1:
                fill = np.zeros((chunk_size * sample_rate) - len(audio_data))
                audio_data = np.concatenate((audio_data, fill))
                previous_audio_data = np.empty(1)
            else:
                # Check if the timestamp_list can safely be merged
                if not (
                    len(audio_data) - max_delta_interval * sample_rate
                    < substract_timestamps(input_timestamp, files, i).seconds
                    < len(audio_data) + max_delta_interval * sample_rate
                ):
                    print(
                        f"Warning: You are trying to merge two audio files that are not chronologically consecutive.\n{files[i-1]} starts at {input_timestamp[input_timestamp['filename'] == files[i-1]]['timestamp'].values[0]} and {files[i]} starts at {input_timestamp[input_timestamp['filename'] == files[i]]['timestamp'].values[0]}."
                    )
                    if (
                        not proceed and sys.__stdin__.isatty()
                    ):  # check if the script runs in an interactive shell. Otherwise will fail if proceed = False
                        res = input(
                            "If you proceed, some timestamps will be lost in the reshaping. Proceed anyway? This message won't show up again if you choose to proceed. ([yes]/no)"
                        )
                        if "yes" in res.lower() or res == "":
                            proceed = True
                        else:
                            sys.exit()
                    elif not proceed and not sys.__stdin__.isatty():
                        print(
                            "Error: Cannot merge non-continuous audio files if force_reshape is false."
                        )
                        sys.exit(1)

                while len(audio_data) < chunk_size * sample_rate and i + 1 < len(files):
                    nextdata, next_sample_rate = sf.read(
                        input_dir_path.joinpath(files[i + 1])
                    )
                    rest = (chunk_size * next_sample_rate) - len(audio_data)
                    audio_data = np.concatenate(
                        (
                            audio_data,
                            nextdata[:rest] if rest <= len(nextdata) else nextdata,
                        )
                    )
                    i += 1
                output = audio_data
                previous_audio_data = nextdata[rest:]

        end_time = (
            (t + 1) * chunk_size
            if chunk_size * sample_rate <= len(output)
            else t * chunk_size + len(output) // sample_rate
        )

        outfilename = output_dir_path.joinpath(
            f"reshaped_from_{t * chunk_size}_to_{end_time}_sec.wav"
        )
        result.append(outfilename.stem)

        timestamp_list.append(
            datetime.strftime(timestamp, "%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        )
        timestamp += timedelta(seconds=chunk_size)

        sf.write(outfilename, output, sample_rate)

        if verbose:
            print(
                f"{outfilename} written! File is {(len(output)/sample_rate)} seconds long. {(len(previous_audio_data)/sample_rate)} seconds left from slicing."
            )
        i += 1
        t += 1

    #! AFTER MAIN LOOP
    while len(previous_audio_data) >= chunk_size * sample_rate:
        output = previous_audio_data[: chunk_size * sample_rate]
        previous_audio_data = previous_audio_data[chunk_size * sample_rate :]
        end_time = (
            (t + 1) * chunk_size
            if chunk_size * sample_rate <= len(output)
            else t * chunk_size + len(output) // sample_rate
        )

        outfilename = output_dir_path.joinpath(
            f"reshaped_from_{t * chunk_size}_to_{end_time}_sec.wav"
        )
        result.append(outfilename.stem)

        timestamp_list.append(
            datetime.strftime(timestamp, "%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        )
        timestamp += timedelta(seconds=chunk_size)

        sf.write(outfilename, output, sample_rate)

        print(
            f"{outfilename} written! File is {(len(output)/sample_rate)} seconds long. {(len(previous_audio_data)/sample_rate)} seconds left from slicing."
        )
        i += 1
        t += 1

    if len(previous_audio_data) > 1:
        skip_last = False
        match last_file_behavior:
            case "truncate":
                output = previous_audio_data
            case "pad":
                fill = np.zeros((chunk_size * sample_rate) - len(previous_audio_data))
                output = np.concatenate((previous_audio_data, fill))
            case "discard":
                skip_last = True

        if not skip_last:
            end_time = t * len(output) // sample_rate

            outfilename = output_dir_path.joinpath(
                f"reshaped_from_{t * chunk_size}_to_{end_time}_sec.wav"
            )
            result.append(outfilename.stem)

            timestamp_list.append(
                datetime.strftime(timestamp, "%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
            )
            timestamp += timedelta(seconds=len(output))

            sf.write(outfilename, output, sample_rate)

            print(
                f"{outfilename} written! File is {(len(output)//sample_rate)} minutes long. {len(previous_audio_data)/sample_rate} minutes left from slicing."
            )

    input_timestamp = pd.DataFrame({"filename": result, "timestamp": timestamp_list})
    input_timestamp.sort_values(by=["timestamp"], inplace=True)
    input_timestamp.to_csv(
        output_dir_path.joinpath("timestamp.csv"),
        index=False,
        na_rep="NaN",
        header=None,
    )

    return [output_dir_path.joinpath(res) for res in result]


if __name__ == "__main__":
    parser = ArgumentParser()
    required = parser.add_argument_group("required arguments")
    required.add_argument(
        "--input-files",
        "-i",
        help="The files to be reshaped, as either the path to a directory containing audio files and a timestamp.csv or a list of filenames all in the same directory alongside a timestamp.csv.",
    )
    required.add_argument(
        "--chunk-size",
        "-s",
        type=int,
        help="The time in seconds of the reshaped files.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        help="The path to the directory to write reshaped files. Default is same as --input-files directory.",
    )
    parser.add_argument(
        "--ind-min",
        "-min",
        type=int,
        default=0,
        help="The first file of the list to be processed. Default is 0.",
    )
    parser.add_argument(
        "--ind-max",
        "-max",
        type=int,
        default=-1,
        help="The last file of the list to be processed. Default is -1, meaning the entire list is processed.",
    )
    parser.add_argument(
        "--offset-beginning",
        type=int,
        default=0,
        help="number of seconds that should be skipped in the first input file. When parallelising the reshaping, it would mean that the beginning of the file is being processed by another job. Default is 0.",
    )
    parser.add_argument(
        "--offset-end",
        type=int,
        default=0,
        help="The number of seconds that should be ignored in the last input file. When parallelising the reshaping, it would mean that the end of this file is processed by another job. Default is 0, meaning that nothing is ignored.",
    )
    parser.add_argument(
        "--max-delta-interval",
        type=int,
        default=5,
        help="The maximum number of second allowed for a delta between two timestamp_list to still be considered the same. Default is 5s up and down.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        default=True,
        help="Whether the script prints informative messages. Default is true.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="If set, deletes all content in --output-dir before writing the output. Default false, deactivated if the --output-dir is the same as --input-file dir.",
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        default=False,
        help="Ignore all warnings and non-fatal errors while reshaping.",
    )

    args = parser.parse_args()

    input_files = (
        args.input_files.split(" ")
        if not Path(args.input_files).is_dir()
        else args.input_files
    )

    files = reshape(
        chunk_size=args.chunk_size,
        input_files=input_files,
        output_dir_path=args.output_dir,
        batch_ind_min=args.batch_ind_min,
        batch_ind_max=args.batch_ind_max,
        offset_beginning=args.offset_beginning,
        offset_end=args.offset_end,
        max_delta_interval=args.max_delta_interval,
        verbose=args.verbose,
        overwrite=args.overwrite,
        force_reshape=args.force,
    )

    if args.verbose:
        print(f"All {len(files)} reshaped audio files written in {files[0].parent}.")
