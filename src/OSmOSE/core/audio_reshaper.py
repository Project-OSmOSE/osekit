import math
import sys
import os
from datetime import datetime, timedelta
from typing import List, Union, Literal
from argparse import ArgumentParser
from pathlib import Path
from filelock import FileLock

import soundfile as sf
import numpy as np
import pandas as pd
from scipy.signal import resample

from OSmOSE.utils import make_path, set_umask, substract_timestamps, from_timestamp, to_timestamp
from OSmOSE.config import *


def reshape(
    input_files: Union[str, list],
    chunk_size: int,
    *,
    new_sr: int = -1,
    output_dir_path: str = None,
    batch_ind_min: int = 0,
    batch_ind_max: int = -1,
    max_delta_interval: int = 5,
    last_file_behavior: Literal["truncate", "pad", "discard"] = "pad",
    offset_beginning: int = 0,
    offset_end: int = 0,
    timestamp_path: Path = None,
    verbose: bool = False,
    overwrite: bool = False,
    force_reshape: bool = False,
    write_output: bool = False,
    merge_files: bool = True
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
        The default method is `pad`.
            The number of seconds that should be skipped in the first input file. When parallelising the reshaping,

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

        merge_files: `bool`, optional, keyword-only
            Whether to merge files when reshaping them. If set to False, then the chunk_size can only be smaller than the original duration, and the remaining
            data will follow the last_file_behavior (default: pad). The default is True.

        write_output: `bool`, optional, keyword-only
            Whether to write the output files or to return the reshaped files in the standard output. The default is False.
    Returns:
    --------
        The list of the path of newly created audio files.
    """
    set_umask()
    verbose = True
    files = []

    if isinstance(input_files, list):
        input_dir_path = Path(input_files[0]).parent
        files = [Path(audio_file).name for audio_file in input_files]
        if verbose:
            print(f"Input directory detected as {input_dir_path}")
    else:
        input_dir_path = Path(input_files)

    #! Validation
    if last_file_behavior not in ["truncate", "pad", "discard"]:
        raise ValueError(
            f"Bad value {last_file_behavior} for last_file_behavior parameters. Must be one of truncate, pad or discard."
        )

    implicit_output = False
    if not output_dir_path:
        print("No output directory provided. Will use the input directory instead.")
        implicit_output = True
        output_dir_path = input_dir_path
        if overwrite:
            print(
                "Cannot overwrite input directory when the output directory is implicit! Choose a different output directory instead."
            )

    output_dir_path = Path(output_dir_path)

    if not input_dir_path.is_dir():
        raise ValueError(
            f"The input files must either be a valid folder path or a list of file path, not {str(input_dir_path)}."
        )

    if not input_dir_path.joinpath("timestamp.csv").exists() and (not timestamp_path or not timestamp_path.exists()):
        raise FileNotFoundError(
            f"The timestamp.csv file must be present in the directory {input_dir_path} and correspond to the audio files in the same location, or be specified in the argument."
        )

    make_path(output_dir_path, mode=DPDEFAULT)

    input_timestamp = pd.read_csv(
        timestamp_path if timestamp_path and timestamp_path.exists() else input_dir_path.joinpath("timestamp.csv"),
        header=None,
        names=["filename", "timestamp",]# "timezone"],
    )

    # When automatically reshaping, will populate the files list
    if not files:
        files = list(
            input_timestamp["filename"][
                batch_ind_min : batch_ind_max + 1
                if batch_ind_max > 0
                else input_timestamp.size
            ]
        )

    if verbose:
        print(f"Files to be reshaped: {','.join(files)}")

    result = []
    timestamp_list = []
    global timestamp # Declare timestamp as a global to use it in yield_result
    timestamp = None
    previous_audio_data = np.empty(0)
    sample_rate = 0
    i = 0
    t = math.ceil(
        sf.info(input_dir_path.joinpath(files[i])).duration
        * (batch_ind_min)
        / chunk_size
    )
    proceed = force_reshape  # Default is False

    def yield_result(*, output, sr, subtype = "FLOAT", extra_text = ""):
        global timestamp
        outfilename = output_dir_path.joinpath(
            f"{from_timestamp(timestamp).replace(':','-').replace('.','_')}.wav"
        )
        result.append(outfilename.name)
        timestamp_list.append(from_timestamp(timestamp))
        timestamp += timedelta(seconds=len(output)//sr)
        
        if write_output:
            sf.write(outfilename, output, sr, format="WAV", subtype=subtype)
            os.chmod(outfilename, mode=FPDEFAULT)

            if verbose:
                if not extra_text:
                    extra_text = f"{len(previous_audio_data)/sr} seconds left from slicing."
                print(
                    f"{outfilename} written! File is {(len(output)/sr)} seconds long. {extra_text}"
                )
        return (output, outfilename)

    while i < len(files):
        with sf.SoundFile(input_dir_path.joinpath(files[i])) as audio_file:
            frames = audio_file.frames
            sample_rate = audio_file.samplerate
            subtype = audio_file.subtype
            audio_data = audio_file.read()

        if new_sr == -1: 
            new_sr = sample_rate
        elif new_sr != sample_rate:
            new_samples = frames*new_sr//sample_rate
            audio_data = resample(audio_data, new_samples)
            sample_rate = new_sr
            
        file_duration = len(audio_data)//sample_rate

        if not merge_files and file_duration < chunk_size:
            raise ValueError("When not merging files, the file duration must be smaller than the target duration.")
        
        if overwrite and not implicit_output and output_dir_path == input_dir_path and output_dir_path == input_dir_path and i<len(files)-1:
            print(f"Deleting {files[i]}")
            input_dir_path.joinpath(files[i]).unlink()

        if i == 0:
            timestamp = input_timestamp[input_timestamp["filename"] == files[i]][
                "timestamp"
            ].values[0]
            timestamp = to_timestamp(timestamp) + timedelta(seconds=offset_beginning)
            audio_data = audio_data[int(offset_beginning * sample_rate) :]
        elif (
            i == len(files) - 1 and offset_end != 0 and not last_file_behavior == "pad"
        ):
            audio_data = audio_data[: int(len(audio_data) - (offset_end * sample_rate))]
        elif previous_audio_data.size <= 1:
            timestamp = to_timestamp(input_timestamp[input_timestamp["filename"] == files[i]][
                "timestamp"
            ].values[0])

        if not merge_files and file_duration < chunk_size:
            raise ValueError("When not merging files, the file duration must be smaller than the target duration.")
        # Need to check if size > 1 because numpy arrays are never empty urgh
        if previous_audio_data.size > 1:
            audio_data = np.concatenate((previous_audio_data, audio_data))
            previous_audio_data = np.empty(0)

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

                yield yield_result(output=output, sr=sample_rate, subtype=subtype)

                t += 1
                audio_data = previous_audio_data

                if not merge_files and len(audio_data) < chunk_size * sample_rate:
                    previous_audio_data = np.empty(0)
                    match last_file_behavior:
                        case "truncate":
                            output = audio_data
                            audio_data = []
                        case "pad":
                            fill = np.zeros((chunk_size * sample_rate) - len(audio_data))
                            output = np.concatenate((audio_data, fill))
                            audio_data = []
                        case "discard":
                            audio_data = []
                            break
                    
                    pad_text = f"Padded with {fill.size // sample_rate} seconds." if last_file_behavior == "pad" and fill.size > 0 else ""
                    print(subtype)
                    yield yield_result(output=output,  sr=sample_rate, subtype=subtype, extra_text=pad_text)



            # If after we get out of the previous while loop we don't have any audio_data left, then we look at the next file.
            if len(audio_data) == 0:
                i += 1
                continue
            
        #! AUDIO DURATION == CHUNK SIZE
        # Else if audio_data is already in the desired duration, output it
        if len(audio_data) == chunk_size * sample_rate:
            output = audio_data
            previous_audio_data = np.empty(0)

        #! AUDIO DURATION < CHUNK_SIZE
        # Else it is shorter, then while the duration is shorter than the desired chunk,
        # we read the next file and append it to the current one.
        elif len(audio_data) < chunk_size * sample_rate:
            # If it is the last file but the audio_data is shorter than the desired chunk, then fill the remaining space with silence.
            if i == len(files) - 1:
                previous_audio_data = audio_data
                break
            else:
                # Check if the timestamp_list can safely be merged
                delta = substract_timestamps(input_timestamp, files, i).seconds - file_duration
                if ( delta > max_delta_interval):
                    print(
                        f"""Warning: You are trying to merge two audio files that are not chronologically consecutive.\n{files[i]} ends at {to_timestamp(input_timestamp[input_timestamp['filename'] == files[i]]['timestamp'].values[0]) + timedelta(seconds=file_duration)} and {files[i+1]} starts at {to_timestamp(input_timestamp[input_timestamp['filename'] == files[i+1]]['timestamp'].values[0])}.\nThere is {delta} seconds of difference between the two files, which is over the maximum tolerance of {max_delta_interval} seconds."""
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
                            # This is meant to close the program with an error while still be user-friendly and test compliant.
                            # Thus we disable the error trace just before raising it to avoid a long trace when the error is clearly identified.
                            sys.tracebacklimit = 0 
                            raise ValueError(
                            "Error: Cannot merge non-continuous audio files if force_reshape is false."
                        )
                    elif not proceed and not sys.__stdin__.isatty():
                        sys.tracebacklimit = 0 
                        raise ValueError(
                            "Error: Cannot merge non-continuous audio files if force_reshape is false."
                        )

                while len(audio_data) < chunk_size * sample_rate and i + 1 < len(files):
                    nextdata, next_sample_rate = sf.read(
                        input_dir_path.joinpath(files[i + 1])
                    )
                    if overwrite and not implicit_output and output_dir_path == input_dir_path and i+1<len(files)-1:
                        print(f"Deleting {files[i+1]}")
                        input_dir_path.joinpath(files[i+1]).unlink()
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

        yield yield_result(output=output, sr=sample_rate, subtype=subtype)


        i += 1
        t += 1

    #! AFTER MAIN LOOP
    while len(previous_audio_data) >= chunk_size * sample_rate:
        output = previous_audio_data[: chunk_size * sample_rate]
        previous_audio_data = previous_audio_data[chunk_size * sample_rate :]

        yield yield_result(output=output, sr=sample_rate, subtype=subtype)

        
        i += 1
        t += 1

    #! REMAINING DATA
    if len(previous_audio_data) > 1:
        skip_last = False
        match last_file_behavior:
            case "truncate":
                output = previous_audio_data
                previous_audio_data = np.empty(0)
            case "pad":
                fill = np.zeros((chunk_size * sample_rate) - len(previous_audio_data))
                output = np.concatenate((previous_audio_data, fill))
                previous_audio_data = np.empty(0)
            case "discard":
                skip_last = True

        if not skip_last:
            yield yield_result(output=output, sr=sample_rate, subtype=subtype)



    for remaining_file in [f for f in files if input_dir_path.joinpath(f).exists()]:
        if overwrite and not implicit_output and output_dir_path == input_dir_path:
            print(f"Deleting {remaining_file}")
            input_dir_path.joinpath(remaining_file).unlink()

    path_csv = output_dir_path.joinpath("timestamp.csv")

    lock = FileLock(str(path_csv) + ".lock")

    with lock:
        # suppr doublons
        if path_csv.exists():
            tmp_timestamp = pd.read_csv(path_csv, header=None)
            result += list(tmp_timestamp[0].values)
            timestamp_list += list(tmp_timestamp[1].values)

        input_timestamp = pd.DataFrame(
            {"filename": result, "timestamp": timestamp_list,}# "timezone": "UTC"}
        )
        input_timestamp.sort_values(by=["timestamp"], inplace=True)
        input_timestamp.drop_duplicates().to_csv(
            path_csv,
            index=False,
            na_rep="NaN",
            header=None,
        )
        os.chmod(path_csv, mode=FPDEFAULT)

    #yield [output_dir_path.joinpath(res) for res in result]

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
        "--batch-ind-min",
        "-min",
        type=int,
        default=0,
        help="The first file of the list to be processed. Default is 0.",
    )
    parser.add_argument(
        "--batch-ind-max",
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
        help="Ignore all warnings and non-fatal errors while reshaping.",
    )
    parser.add_argument(
        "--last-file-behavior",
        default="pad",
        help="Tells the program what to do with the remaining data that are shorter than the chunk size. Possible arguments are pad (the default), which pads with silence until the last file has the same length as the others; truncate to create a shorter file with only the leftover data; discard to not do anything with the last data and throw it away.",
    )
    parser.add_argument(
        "--timestamp-path",
        default=None,
        help="Path to the original timestamp file."
    )
    parser.add_argument(
        "--no-merge",
        action="store_false",
        help="Don't try to merge the reshaped files."
    ) # When absent = we merge file; when present = we don't merge -> merge_file is False

    args = parser.parse_args()

    input_files = (
        args.input_files.split(" ")
        if not Path(args.input_files).is_dir()
        else args.input_files
    )

    print("Parameters :", args)

    files = reshape(
        chunk_size=args.chunk_size,
        input_files=input_files,
        output_dir_path=args.output_dir,
        batch_ind_min=args.batch_ind_min,
        batch_ind_max=args.batch_ind_max,
        offset_beginning=args.offset_beginning,
        offset_end=args.offset_end,
        timestamp_path=Path(args.timestamp_path),
        max_delta_interval=args.max_delta_interval,
        last_file_behavior=args.last_file_behavior,
        verbose=args.verbose,
        overwrite=args.overwrite,
        force_reshape=args.force,
        merge_files=args.no_merge
    )

    if args.verbose:
        print(f"All {len(files)} reshaped audio files written in {files[0].parent}.")
