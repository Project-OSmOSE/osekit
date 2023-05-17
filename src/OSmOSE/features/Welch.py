from datetime import datetime, timedelta
from functools import partial
import inspect
import os
import shutil
import sys
from typing import List, Tuple, Union, Literal
from math import log10
from pathlib import Path
import multiprocessing as mp
from filelock import FileLock

import soundfile as sf
import pandas as pd
import numpy as np
from scipy import signal
from termcolor import colored
from matplotlib import pyplot as plt
from OSmOSE.job import Job_builder
from OSmOSE.cluster import (
    reshape,
    resample,
    compute_stats,
)
from OSmOSE.Dataset import Dataset
from OSmOSE.utils import safe_read, make_path, set_umask, from_timestamp, to_timestamp
from OSmOSE.config import *


class Welch(Dataset):
    """Main class for spectrogram-related computations. Can resample, reshape and normalize audio files before generating spectrograms."""

    def __init__(
        self,
        dataset_path: str,
        *,
        dataset_sr: int = None,
        gps_coordinates: Union[str, list, tuple] = None,
        owner_group: str = None,
        analysis_params: dict = None,
        batch_number: int = 10,
        local: bool = False,
    ) -> None:
        """Instanciates a spectrogram object.

        The characteristics of the dataset are essential to input for the generation of the spectrograms. There is three ways to input them:
            - Use the existing `analysis/analysis_sheet.csv` file. If one exist, it will take priority over the other methods. Note that
            when using this file, some attributes will be locked in read-only mode.
            - Fill the `analysis_params` argument. More info on the expected value below.
            - Don't initialize the attributes in the constructor, and assign their values manually.

        In any case, all attributes must have a value for the spectrograms to be generated. If it does not exist, `analysis/analysis_sheet.csv`
        will be written at the end of the `Spectrogram.initialize()` method.

        Parameters
        ----------
        dataset_path : `str`
            The absolute path to the dataset folder. The last folder in the path will be considered as the name of the dataset.
        dataset_sr : `int`, keyword-only
            The sample rate used for the generation of the spectrograms.
        coordinates : `str` or `list` or `tuple`, optional, keyword-only
            The GPS coordinates of the listening location. If it is of type `str`, it must be the name of a csv file located in `raw/auxiliary`,
            otherwise a list or a tuple with the first element being the latitude coordinates and second the longitude coordinates.
        osmose_group_name : `str`, optional, keyword-only
            The name of the group using the OsmOSE package. All files created using this dataset will be accessible by the osmose group.
            Will not work on Windows.
        analysis_params : `dict`, optional, keyword-only
            If `analysis/analysis_sheet.csv` does not exist, the analysis parameters can be submitted in the form of a dict,
            with keys matching what is expected:
                - nfft : `int`
                - window_size : `int`
                - overlap : `int`
                - colormap : `str`
                - zoom_level : `int`
                - dynamic_min : `int`
                - dynamic_max : `int`
                - number_adjustment_spectrogram : `int`
                - spectro_duration : `int`
                - zscore_duration : `float` or `str`
                - hp_filter_min_freq : `int`
                - sensitivity_dB : `int`
                - peak_voltage : `float`
                - spectro_normalization : `str`
                - data_normalization : `str`
                - gain_dB : `int`
            If additional information is given, it will be ignored. Note that if there is an `analysis/analysis_sheet.csv` file, it will
            always have the priority.
        batch_number : `int`, optional, keyword_only
            The number of batches the dataset files will be split into when submitting parallel jobs (the default is 10).
        local : `bool`, optional, keyword_only
            Indicates whether or not the program is run locally. If it is the case, it will not create jobs and will handle the paralelisation
            alone. The default is False.
        """
        super().__init__(
            dataset_path=dataset_path,
            gps_coordinates=gps_coordinates,
            owner_group=owner_group,
        )

        self.__local = local

        if self.is_built:
            orig_metadata = pd.read_csv(self._get_original_after_build().joinpath("metadata.csv"), header=0)
        elif not dataset_sr:
            raise ValueError('If you dont know your sr, please use the build() method first')
        processed_path = self.path.joinpath(OSMOSE_PATH.spectrogram)
        metadata_path = processed_path.joinpath("adjust_metadata.csv")
        if metadata_path.exists():
            self.__analysis_file = True
            analysis_sheet = pd.read_csv(metadata_path, header=0)
        elif analysis_params:
            self.__analysis_file = False
            # We put the value in a list so that value[0] returns the right value below.
            analysis_sheet = {key: [value] for (key, value) in analysis_params.items()}
        else:
            analysis_sheet = {}
            self.__analysis_file = False
            print(
                "No valid processed/adjust_metadata.csv found and no parameters provided. All attributes will be initialized to default values..  \n"
            )

        self.batch_number: int = batch_number
        self.dataset_sr: int = dataset_sr if dataset_sr is not None else orig_metadata['origin_sr'][0]

        self.nfft: int = analysis_sheet["nfft"][0] if "nfft" in analysis_sheet else 1
        self.window_size: int = (
            analysis_sheet["window_size"][0]
            if "window_size" in analysis_sheet
            else None
        )
        self.overlap: int = (
            analysis_sheet["overlap"][0] if "overlap" in analysis_sheet else None
        )
        self.colormap: str = (
            analysis_sheet["colormap"][0] if "colormap" in analysis_sheet else "viridis"

        )
        self.zoom_level: int = (
            analysis_sheet["zoom_level"][0] if "zoom_level" in analysis_sheet else 0
        )
        self.dynamic_min: int = (
            analysis_sheet["dynamic_min"][0]
            if "dynamic_min" in analysis_sheet
            else None
        )
        self.dynamic_max: int = (
            analysis_sheet["dynamic_max"][0]
            if "dynamic_max" in analysis_sheet
            else None
        )
        self.number_adjustment_spectrogram: int = (
            analysis_sheet["number_adjustment_spectrogram"][0]
            if "number_adjustment_spectrogram" in analysis_sheet
            else None
        )
        self.spectro_duration: int = (
            analysis_sheet["spectro_duration"][0]
            if analysis_sheet is not None and "spectro_duration" in analysis_sheet
            else (
                orig_metadata["audio_file_origin_duration"][0]
                if self.is_built
                else -1
            )
        )

        self.zscore_duration: Union[float, str] = (
            analysis_sheet["zscore_duration"][0]
            if "zscore_duration" in analysis_sheet
            and isinstance(analysis_sheet["zscore_duration"][0], float)
            else "original"
        )

        # fmin cannot be 0 in butterworth. If that is the case, it takes the smallest value possible, epsilon
        self.hp_filter_min_freq: int = (
            analysis_sheet["hp_filter_min_freq"][0]
            if "hp_filter_min_freq" in analysis_sheet
            else 0
        )

        self.sensitivity: float = (
            analysis_sheet["sensitivity_dB"][0]
            if "sensitivity_dB" in analysis_sheet
            else 0
        )

        self.peak_voltage: float = (
            analysis_sheet["peak_voltage"][0]
            if "peak_voltage" in analysis_sheet
            else None
        )
        self.spectro_normalization: str = (
            analysis_sheet["spectro_normalization"][0]
            if "spectro_normalization" in analysis_sheet
            else None
        )
        self.data_normalization: str = (
            analysis_sheet["data_normalization"][0]
            if "data_normalization" in analysis_sheet
            else None
        )
        self.gain_dB: float = (
            analysis_sheet["gain_dB"][0]
            if "gain_dB" in analysis_sheet is not None
            else None
        )

        self.window_type: str = (
            analysis_sheet["window_type"][0]
            if "window_type" in analysis_sheet
            else "hamming"
        )

        self.time_resolution = (
            [
                analysis_sheet[col][0]
                for col in analysis_sheet
                if "time_resolution" in col
            ]
            if analysis_sheet is not None
            else [0] * self.zoom_level
        )

        self.jb = Job_builder()

        plt.switch_backend("agg")

        fontsize = 16
        ticksize = 12
        plt.rc("font", size=fontsize)  # controls default text sizes
        plt.rc("axes", titlesize=fontsize)  # fontsize of the axes title
        plt.rc("axes", labelsize=fontsize)  # fontsize of the x and y labels
        plt.rc("xtick", labelsize=ticksize)  # fontsize of the tick labels
        plt.rc("ytick", labelsize=ticksize)  # fontsize of the tick labels
        plt.rc("legend", fontsize=ticksize)  # legend fontsize
        plt.rc("figure", titlesize=ticksize)  # fontsize of the figure title
        self.adjust = False
        self.__build_path(dry=True)

    # region Spectrogram properties

    @property
    def dataset_sr(self):
        """int: The sampling frequency of the dataset."""
        return self.__dataset_sr

    @dataset_sr.setter
    def dataset_sr(self, value: int):
        self.__dataset_sr = value

    @property
    def nfft(self):
        """int: The number of Fast Fourier Transform used to generate the spectrograms."""
        return self.__nfft

    @nfft.setter
    def nfft(self, value: int):
        self.__nfft = value

    @property
    def window_size(self):
        """int: The window size of the generated spectrograms."""
        return self.__window_size

    @window_size.setter
    def window_size(self, value: int):
        self.__window_size = value

    @property
    def overlap(self):
        """int: The overlap percentage between two spectrogram windows."""
        return self.__overlap

    @overlap.setter
    def overlap(self, value: int):
        self.__overlap = value

    @property
    def colormap(self):
        """str: The type of colormap of the spectrograms."""
        return self.__colormap

    @colormap.setter
    def colormap(self, value: str):
        self.__colormap = value

    @property
    def zoom_level(self):
        """int: Number of zoom levels."""
        return self.__zoom_level

    @zoom_level.setter
    def zoom_level(self, value: int):
        self.__zoom_level = value

    @property
    def dynamic_min(self):
        """int: Minimum value of the colormap."""
        return self.__dynamic_min

    @dynamic_min.setter
    def dynamic_min(self, value: int):
        self.__dynamic_min = value

    @property
    def dynamic_max(self):
        """int: Maximum value of the colormap."""
        return self.__dynamic_max

    @dynamic_max.setter
    def dynamic_max(self, value: int):
        self.__dynamic_max = value

    @property
    def number_adjustment_spectrogram(self):
        """int: Number of spectrograms used to adjust the parameters."""
        return self.__number_adjustment_spectrogram

    @number_adjustment_spectrogram.setter
    def number_adjustment_spectrogram(self, value: int):
        self.__number_adjustment_spectrogram = value

    @property
    def spectro_duration(self):
        """int: Duration of the spectrogram (at the lowest zoom level) in seconds."""
        return self.__spectro_duration

    @spectro_duration.setter
    def spectro_duration(self, value: int):
        self.__spectro_duration = value

    @property
    def zscore_duration(self):
        return self.__zscore_duration

    @zscore_duration.setter
    def zscore_duration(self, value: int):
        self.__zscore_duration = value

    @property
    def hp_filter_min_freq(self):
        """float: Floor frequency for the High Pass Filter."""
        return self.__hp_filter_min_freq

    @hp_filter_min_freq.setter
    def hp_filter_min_freq(self, value:int):
        self.__hp_filter_min_freq = value

    @property
    def sensitivity(self):
        """int: Numeric sensitivity of the recording device."""
        return self.__sensitivity

    @sensitivity.setter
    def sensitivity(self, value):
        """Always assume the sensitivity is given in dB."""
        self.__sensitivity = 10 ** (value / 20) * 1e6

    @property
    def peak_voltage(self):
        """float: The maximum voltage of the device."""
        return self.__peak_voltage

    @peak_voltage.setter
    def peak_voltage(self, value: float):
        self.__peak_voltage = value

    @property
    def spectro_normalization(self):
        """str: Type of normalization used to generate the spectrograms."""
        return self.__spectro_normalization

    @spectro_normalization.setter
    def spectro_normalization(self, value: Literal["spectrum", "density"]):
        self.__spectro_normalization = value

    @property
    def data_normalization(self):
        """str: Type of normalization applied to the data."""
        return self.__data_normalization

    @data_normalization.setter
    def data_normalization(self, value: Literal["instrument", "zscore"]):
        self.__data_normalization = value

    @property
    def gain_dB(self):
        """float: Gain of the device in decibels."""
        return self.__gain_dB

    @gain_dB.setter
    def gain_dB(self, value: float):
        self.__gain_dB = value

    @property
    def window_type(self):
        """str: Type of the window used to generate the spectrograms."""
        return self.__window_type

    @window_type.setter
    def window_type(self, value: Literal["hamming"]):
        self.__window_type = value

    @property
    def frequency_resolution(self) -> float:
        """Frequency resolution of the spectrogram, calculated by dividing the sample rate by the number of nfft."""
        return self.dataset_sr / self.nfft

    @property
    def time_resolution(self):
        return self.__time_resolution

    @time_resolution.setter
    def time_resolution(self, value):
        self.__time_resolution = value

    # endregion

    def __build_path(self, adjust: bool = False, dry: bool = False):
        """Build some internal paths according to the expected architecture and might create them.

        Parameter
        ---------
            adjust : `bool`, optional
                Whether or not the paths are used to adjust spectrogram parameters.
            dry: `bool`, optional
                If set to True, will not create the folders and just return the file path.
        """
        set_umask()
        processed_path = self.path.joinpath(OSMOSE_PATH.spectrogram)
        audio_foldername = f"{str(self.spectro_duration)}_{str(self.dataset_sr)}"
        self.audio_path = self.path.joinpath(OSMOSE_PATH.raw_audio, audio_foldername)

        if adjust:
            self.__spectro_foldername = "adjustment_spectros"
        else:
            self.__spectro_foldername = (
                f"{str(self.nfft)}_{str(self.window_size)}_{str(self.overlap)}"
            )

        self.path_output_spectrogram = processed_path.joinpath(
            audio_foldername, self.__spectro_foldername, "image"
        )

        self.path_output_spectrogram_matrix = processed_path.joinpath(
            audio_foldername, self.__spectro_foldername, "matrix"
        )

        # Create paths
        if not dry:
            make_path(self.audio_path, mode=DPDEFAULT)
            make_path(self.path_output_spectrogram, mode=DPDEFAULT)
            if not adjust:
                make_path(self.path_output_spectrogram_matrix, mode=DPDEFAULT)
                make_path(self.path.joinpath(OSMOSE_PATH.statistics), mode=DPDEFAULT)

    def check_spectro_size(self):
        """Verify if the parameters will generate a spectrogram that can fit one screen properly"""
        if self.nfft > 2048:
            print("your nfft is :", self.nfft)
            print(
                colored(
                    "PLEASE REDUCE IT UNLESS YOU HAVE A VERY HD SCREEN WITH MORE THAN 1k pixels vertically !!!! ",
                    "red",
                )
            )

        tile_duration = self.spectro_duration / 2 ** (self.zoom_level)

        data = np.zeros([int(tile_duration * self.dataset_sr), 1])

        Noverlap = int(self.window_size * self.overlap / 100)

        Nbech = np.size(data)
        Noffset = self.window_size - Noverlap
        Nbwin = int((Nbech - self.window_size) / Noffset)
        Freq = np.fft.rfftfreq(self.nfft, d=1 / self.dataset_sr)
        Time = np.linspace(0, Nbech / self.dataset_sr, Nbwin)

        print("your smallest tile has a duration of:", tile_duration, "(s)")
        print("\n")

        if Nbwin > 3500:
            print(
                colored(
                    "PLEASE REDUCE IT UNLESS YOU HAVE A VERY HD SCREEN WITH MORE THAN 2k pixels horizontally !!!! ",
                    "red",
                )
            )

        print("\n")
        print("your number of time windows in this tile is:", Nbwin)
        print("\n")
        print(
            "your resolutions : time = ",
            round(Time[1] - Time[0], 3),
            "(s) / frequency = ",
            round(Freq[1] - Freq[0], 3),
            "(Hz)",
        )

    # TODO: some cleaning
    def initialize(
        self,
        *,
        dataset_sr: int = None,
        reshape_method: Literal["legacy", "classic", "none"] = "none",
        batch_ind_min: int = 0,
        batch_ind_max: int = -1,
        pad_silence: bool = False,
        force_init: bool = False,
        date_template: str = None,
        merge_on_reshape: bool = True,
        last_file_behavior: Literal["pad","truncate","discard"] = "pad"
    ) -> None:
        """Prepares everything (path, variables, files) for spectrogram generation. This needs to be run before the spectrograms are generated.
        If the dataset has not yet been build, it is before the rest of the functions are initialized.

        Parameters
        ----------
        dataset_sr : `int`, optional, keyword-only
            The sampling frequency of the audio files used to generate the spectrograms. If set, will overwrite the Spectrogram.dataset_sr attribute.
        reshape_method : {"legacy", "classic", "none"}, optional, keyword-only
            Which method to use if the desired size of the spectrogram is different from the audio file duration.
            - legacy : Legacy method, use bash and sox software to trim the audio files and fill the empty space with nothing.
            Unpractical when the audio file duration is longer than the desired spectrogram size.
            - classic : Classic method, use python and sox library to cut and concatenate the audio files to fit the desired duration.
            Will rewrite the `timestamp.csv` file, thus timestamps may have unexpected behavior if the concatenated files are not chronologically
            subsequent.
            - none : Don't reshape, will throw an error if the file duration is different than the desired spectrogram size. (It is the default behavior)

        batch_ind_min : `int`, optional, keyword-only
            The index of the first file to consider. Both this parameter and `batch_ind_max` are not commonly used and are
            for very specific use cases. Most of the time, you want to initialize the whole dataset (the default is 0).
        batch_ind_max : `int`, optional, keyword-only
            The index of the last file to consider (the default is -1, meaning consider every file).
        pad_silence : `bool`, optional, keyword-only
            When using the legacy reshaping method, whether there should be a silence padding or not (default is False).
        force_init : `bool`, optional, keyword-only
            Force every parameter of the initialization.
        date_template : `str`, optiona, keyword-only
            When initializing a spectrogram of a dataset that has not been built, providing a date_template will generate the timestamp.csv.
        """
        # Mandatory init
        if not self.is_built:
            try:
                self.build(date_template=date_template)
            except Exception as e:
                print(
                    f"Unhandled error during dataset building. The spectrogram initialization will be cancelled. The error may be resolved by building the dataset separately first. Description of the error: {str(e)}"
                )
                return

        self.__build_path()

        if dataset_sr:
            self.dataset_sr = dataset_sr

        self.path_input_audio_file = self._get_original_after_build()
        list_wav_withEvent_comp = sorted(self.path_input_audio_file.glob(f"*.({'|'.join(SUPPORTED_AUDIO_FORMAT)})"))

        if batch_ind_max == -1:
            batch_ind_max = len(list_wav_withEvent_comp)
        list_wav_withEvent = list_wav_withEvent_comp[batch_ind_min:batch_ind_max]

        self.list_wav_to_process = [
            audio_file.name for audio_file in list_wav_withEvent
        ]

        if self.data_normalization == "zscore" and self.spectro_normalization != "spectrum":
            self.spectro_normalization = "spectrum"
            print("WARNING: the spectrogram normalization has been changed to spectrum because the data will be normalized using zscore.")

        """List containing the last job ids to grab outside of the class."""
        self.pending_jobs = []

        # Stop initialization if already done
        final_path = self.path.joinpath(
            OSMOSE_PATH.spectrogram,
            f"{str(self.spectro_duration)}_{str(self.dataset_sr)}",
            f"{str(self.nfft)}_{str(self.window_size)}_{str(self.overlap)}",
            "metadata.csv",
        )
        temp_path = self.path.joinpath(OSMOSE_PATH.spectrogram, "adjust_metadata.csv")
        audio_metadata_path = self.path.joinpath(
            OSMOSE_PATH.raw_audio,
            f"{str(self.spectro_duration)}_{str(self.dataset_sr)}",
            "metadata.csv",
        )

        if (
            (final_path.exists() or temp_path.exists())
            and audio_metadata_path.exists()
            and audio_metadata_path.with_stem("timestamp").exists()
            and not force_init
        ):
            audio_file_count = pd.read_csv(audio_metadata_path)["audio_file_count"][0]
            if len(list(audio_metadata_path.parent.glob(f"*.({'|'.join(SUPPORTED_AUDIO_FORMAT)})")) == audio_file_count):
                print(
                    "It seems these spectrogram parameters are already initialized. If it is an error or you want to rerun the initialization, add the `force_init` argument."
                )
                return

        #! INITIALIZATION START
        # Load variables from raw metadata
        metadata = pd.read_csv(self.path_input_audio_file.joinpath("metadata.csv"))
        audio_file_origin_duration = metadata["audio_file_origin_duration"][0]
        sr_origin = metadata["dataset_sr"][0]
        audio_file_count = metadata["audio_file_count"][0]

        if self.path.joinpath(OSMOSE_PATH.processed, "subset_files.csv").is_file():
            subset = pd.read_csv(
                self.path.joinpath(OSMOSE_PATH.processed, "subset_files.csv"),
                header=None,
            )[0].values
            self.list_wav_to_process = list(
                set(subset).intersection(set(self.list_wav_to_process))
            )

        # Generate the timestamp.csv
        input_timestamp = pd.read_csv(
            self.path_input_audio_file.joinpath("timestamp.csv"),
            header=None,
            names=["filename", "timestamp", "timezone"],
        )

        new_timestamp_list = []
        new_name_list = []
        for i in range(len(input_timestamp.filename)):
            if len(new_timestamp_list) == 0:
                next_timestamp = to_timestamp(input_timestamp.timestamp[0])

            new_timestamp_list.append(from_timestamp(next_timestamp))
            new_name_list.append(f"{new_timestamp_list[-1].replace(':','-').replace('.','_')}.wav")
            
            next_timestamp += timedelta(seconds=self.spectro_duration)

        path_csv = self.audio_path.joinpath("timestamp.csv")

        if self.path_input_audio_file == self.audio_path:
            path_csv.rename(path_csv.with_stem("old_timestamp"))

        new_timestamp = pd.DataFrame(
            {"filename": new_name_list, "timestamp": new_timestamp_list, "timezone": "UTC"}
        )
        new_timestamp.sort_values(by=["timestamp"], inplace=True)
        new_timestamp.drop_duplicates().to_csv(
            path_csv,
            index=False,
            na_rep="NaN",
            header=None,
        )
        os.chmod(path_csv, mode=FPDEFAULT)



        batch_size = len(self.list_wav_to_process) // self.batch_number

        #! RESAMPLING
        resample_job_id_list = []
        processes = []
        resample_done = False

        if self.dataset_sr != sr_origin and (not os.listdir(self.audio_path) or force_init):
            resample_done = True
            for batch in range(self.batch_number):
                i_min = batch * batch_size
                i_max = (
                    i_min + batch_size
                    if batch < self.batch_number - 1
                    else len(self.list_wav_to_process)
                )  # If it is the last batch, take all files

                if self.__local:
                    process = mp.Process(
                        target=resample,
                        kwargs={
                            "input_dir": self.path_input_audio_file,
                            "output_dir": self.audio_path,
                            "target_sr": self.dataset_sr,
                            "batch_ind_min": i_min,
                            "batch_ind_max": i_max,
                        },
                    )

                    process.start()
                    processes.append(process)
                else:
                    self.jb.build_job_file(
                        script_path=Path(inspect.getfile(resample)).resolve(),
                        script_args=f"--input-dir {self.path_input_audio_file} --target-sr {self.dataset_sr} --batch-ind-min {i_min} --batch-ind-max {i_max} --output-dir {self.audio_path}",
                        jobname="OSmOSE_resample",
                        preset="low",
                        logdir=self.path.joinpath("log")
                    )
                    # TODO: use importlib.resources

                    job_id = self.jb.submit_job()
                    resample_job_id_list += job_id

            self.pending_jobs = resample_job_id_list
            for process in processes:
                process.join()

        #! ZSCORE NORMALIZATION
        norma_job_id_list = []
        if (
            #os.listdir(self.path.joinpath(OSMOSE_PATH.statistics))
            self.data_normalization == "zscore"
            and self.zscore_duration is not None
            and len(os.listdir(self.path.joinpath(OSMOSE_PATH.statistics))) == 0
            or force_init
        ):
            shutil.rmtree(self.path.joinpath(OSMOSE_PATH.statistics), ignore_errors=True)
            make_path(self.path.joinpath(OSMOSE_PATH.statistics), mode=DPDEFAULT)
            for batch in range(self.batch_number):
                i_min = batch * batch_size
                i_max = (
                    i_min + batch_size
                    if batch < self.batch_number - 1
                    else len(self.list_wav_to_process)
                )  # If it is the last batch, take all files
                if self.__local:
                    process = mp.Process(
                        target=compute_stats,
                        kwargs={
                            "input_dir": self.path_input_audio_file,
                            "output_file": self.path.joinpath(
                                OSMOSE_PATH.statistics,
                                "SummaryStats_" + str(i_min) + ".csv",
                            ),
                            "target_sr": self.dataset_sr,
                            "batch_ind_min": i_min,
                            "batch_ind_max": i_max,
                        },
                    )

                    process.start()
                    processes.append(process)
                else:
                    jobfile = self.jb.build_job_file(
                        script_path=Path(inspect.getfile(compute_stats)).resolve(),
                        script_args=f"--input-dir {self.path_input_audio_file} --hp-filter-min-freq {self.hp_filter_min_freq} \
                                    --batch-ind-min {i_min} --batch-ind-max {i_max} --output-file {self.path.joinpath(OSMOSE_PATH.statistics, 'SummaryStats_' + str(i_min) + '.csv')}",
                        jobname="OSmOSE_get_zscore_params",
                        preset="low",
                        logdir=self.path.joinpath("log")
                    )

                    job_id = self.jb.submit_job(dependency=resample_job_id_list)
                    norma_job_id_list += job_id
            
            self.pending_jobs = norma_job_id_list

            for process in processes:
                process.join()

        #! RESHAPING
        # Reshape audio files to fit the maximum spectrogram size, whether it is greater or smaller.
        reshape_job_id_list = []

        if self.spectro_duration != int(audio_file_origin_duration):
            # We might reshape the files and create the folder. Note: reshape function might be memory-heavy and deserve a proper qsub job.
            if self.spectro_duration > int(
                audio_file_origin_duration
            ) and reshape_method in ["none", "legacy"]:
                raise ValueError(
                    "Spectrogram size cannot be greater than file duration. If you want to automatically reshape your audio files to fit the spectrogram size, consider setting the reshape method to 'reshape'."
                )

            print(
                f"Automatically reshaping audio files to fit the spectro duration value. Files will be {self.spectro_duration} seconds long."
            )

            if reshape_method == "classic":
                # build job, qsub, stuff

                input_files = self.audio_path if resample_done else self.path_input_audio_file

                nb_reshaped_files = (
                    audio_file_origin_duration * audio_file_count
                ) / self.spectro_duration
                metadata["audio_file_count"] = nb_reshaped_files
                next_offset_beginning = 0
                offset_end = 0
                i_max = -1

                for batch in range(self.batch_number):
                    
                    if i_max >= len(self.list_wav_to_process) - 1:
                        continue

                    offset_beginning = next_offset_beginning
                    next_offset_beginning = 0

                    i_min = i_max + (1 if not offset_beginning else 0)
                    i_max = (
                        i_min + batch_size
                        if batch < self.batch_number - 1
                        and i_min + batch_size < len(self.list_wav_to_process)
                        else len(self.list_wav_to_process) - 1
                    )  # If it is the last batch, take all files

                    if merge_on_reshape:
                        while (
                            (
                                (i_max - i_min + 1) * audio_file_origin_duration
                                - offset_end
                                - offset_beginning  # Determines if the offset would require more than one file
                            )
                            % self.spectro_duration
                            > audio_file_origin_duration
                            and i_max < len(self.list_wav_to_process)
                        ) or (
                            i_max - i_min + offset_end - offset_beginning + 1
                        ) * audio_file_origin_duration < self.spectro_duration:
                            i_max += 1

                        last_file_behavior = (
                            "pad"
                            if batch == self.batch_number - 1
                            or i_max == len(self.list_wav_to_process) - 1
                            else "discard"
                        )

                        offset_end = (
                            (i_max - i_min + 1) * audio_file_origin_duration
                            - offset_beginning
                        ) % self.spectro_duration
                        if offset_end:
                            next_offset_beginning = audio_file_origin_duration - offset_end
                        else:
                            offset_end = 0  # ? ack

                    if self.__local:
                        process = mp.Process(
                            target=reshape,
                            kwargs={
                                "input_files": input_files,
                                "chunk_size": self.spectro_duration,
                                "output_dir_path": self.audio_path,
                                "offset_beginning": int(offset_beginning),
                                "offset_end": int(offset_end),
                                "batch_ind_min": i_min,
                                "batch_ind_max": i_max,
                                "last_file_behavior": last_file_behavior,
                                "timestamp_path": self.path_input_audio_file.joinpath("timestamp.csv"),
                                "merge_files": merge_on_reshape
                            },
                        )

                        process.start()
                        processes.append(process)
                    else:
                        self.jb.build_job_file(
                            script_path=Path(inspect.getfile(reshape)).resolve(),
                            script_args=f"--input-files {input_files} --chunk-size {self.spectro_duration} --batch-ind-min {i_min}\
                                        --batch-ind-max {i_max} --output-dir {self.audio_path} --timestamp-path {self.path_input_audio_file.joinpath('timestamp.csv')}\
                                        --offset-beginning {int(offset_beginning)} --offset-end {int(offset_end)}\
                                        --last-file-behavior {last_file_behavior} {'--force' if force_init else ''}\
                                        {'--overwrite' if resample_done else ''} {'--no-merge' if not merge_on_reshape else ''}",
                            jobname="OSmOSE_reshape_py",
                            preset="low",
                            mem="30G",
                            walltime="04:00:00",
                            logdir=self.path.joinpath("log")
                        )

                        job_id = self.jb.submit_job(dependency=norma_job_id_list if norma_job_id_list else resample_job_id_list)
                        reshape_job_id_list += job_id
                self.pending_jobs = reshape_job_id_list

                for process in processes:
                    process.join()

        
        metadata["dataset_fileDuration"] = self.spectro_duration
        new_meta_path = self.audio_path.joinpath("metadata.csv")
        metadata.to_csv(new_meta_path)
        os.chmod(new_meta_path, mode=FPDEFAULT)

        data = {
            "dataset_name": self.name,
            "dataset_sr": self.dataset_sr,
            "nfft": self.nfft,
            "window_size": self.window_size,
            "overlap": self.overlap,
            "colormap": self.colormap,
            "zoom_level": self.zoom_level,
            "number_adjustment_spectrogram": self.number_adjustment_spectrogram,
            "dynamic_min": self.dynamic_min,
            "dynamic_max": self.dynamic_max,
            "spectro_duration": self.spectro_duration,
            "audio_file_folder_name": self.audio_path.name,
            "data_normalization": self.data_normalization,
            "hp_filter_min_freq": self.hp_filter_min_freq,
            "sensitivity_dB": 20 * log10(self.sensitivity / 1e6),
            "peak_voltage": self.peak_voltage,
            "spectro_normalization": self.spectro_normalization,
            "gain_dB": self.gain_dB,
            "zscore_duration": self.zscore_duration,
            "window_type": self.window_type,
            "frequency_resolution": self.frequency_resolution,
        }

        for i, time_res in enumerate(self.time_resolution):
            data.update({f"time_resolution_{i}": time_res})

        analysis_sheet = pd.DataFrame.from_records([data])

        adjust_path = self.path.joinpath(OSMOSE_PATH.spectrogram, "adjust_metadata.csv")
        if adjust_path.exists():
            adjust_path.unlink() # Always overwrite this file

        analysis_sheet.to_csv(
            adjust_path
        )
        os.chmod(adjust_path, mode=FPDEFAULT)
        
        if not adjust_path.exists():
            raise FileNotFoundError("Failed to write adjust_metadata.csv")

    def audio_file_list_csv(self) -> Path:
        list_audio = list(self.audio_path.glob(f"*.({'|'.join(SUPPORTED_AUDIO_FORMAT)})"))
        csv_path = self.audio_path.joinpath(f"wav_list_{len(list_audio)}.csv")

        if csv_path.exists():
            return csv_path
        else:
            with open(csv_path, "w") as f:
                f.write("\n".join([str(audio) for audio in list_audio]))

            os.chmod(csv_path, mode=FPDEFAULT)

            return csv_path

    def update_parameters(self, filename: Path) -> bool:
        """Read the csv file filename and compare it to the spectrogram parameters. If any parameter is different, the file will be replaced and the fields changed.
        
        If there is nothing to update, the file won't be changed.
        
        Parameter
        ---------
        filename: Path
            The path to the csv file to be updated.
        
        Returns
        -------
            True if the parameters were updated else False."""
        
        if not filename.suffix == ".csv":
            raise ValueError("The file must be a .csv file to be updated.")
        new_params = {
            "dataset_name": self.name,
            "dataset_sr": self.dataset_sr,
            "nfft": self.nfft,
            "window_size": self.window_size,
            "overlap": self.overlap,
            "colormap": self.colormap,
            "zoom_level": self.zoom_level,
            "number_adjustment_spectrogram": self.number_adjustment_spectrogram,
            "dynamic_min": self.dynamic_min,
            "dynamic_max": self.dynamic_max,
            "spectro_duration": self.spectro_duration,
            "audio_file_folder_name": self.audio_path.name,
            "data_normalization": self.data_normalization,
            "hp_filter_min_freq": self.hp_filter_min_freq,
            "sensitivity_dB": 20 * log10(self.sensitivity / 1e6),
            "peak_voltage": self.peak_voltage,
            "spectro_normalization": self.spectro_normalization,
            "gain_dB": self.gain_dB,
            "zscore_duration": self.zscore_duration,
            "window_type": self.window_type,
            "frequency_resolution": self.frequency_resolution,
        }
        for i, time_res in enumerate(self.time_resolution):
            new_params.update({f"time_resolution_{i}": time_res})

        if not filename.exists():
            pd.DataFrame.from_records([new_params]).to_csv(filename)

            os.chmod(filename, mode=DPDEFAULT)
            return True
        
        orig_params = pd.read_csv(filename)
        
        if any([param not in orig_params or str(orig_params[param]) != str(new_params[param]) for param in new_params]):
            filename.unlink()
            pd.DataFrame.from_records([new_params]).to_csv(filename)

            os.chmod(filename, mode=DPDEFAULT)
            return True
        return False

    def to_csv(self, filename: Path) -> None:
        """Outputs the characteristics of the spectrogram the specified file in csv format.

        Parameter
        ---------
        filename: Path
            The path of the file to be written."""

        data = {
            "name": self.__spectro_foldername,
            "nfft": self.nfft,
            "window_size": self.window_size,
            "overlap": self.overlap / 100,
            "zoom_level": 2 ** (self.zoom_level - 1),
            # "dynamic_min": self.dynamic_min,
            # "dynamic_max": self.dynamic_max,
            # "number_adjustment_spectrogram": self.number_adjustment_spectrogram,
            # "spectro_duration": self.spectro_duration,
            # "zscore_duration": self.zscore_duration,
            # "hp_filter_min_freq": self.hp_filter_min_freq,
            # "sensitivity_dB": 20 * log10(self.sensitivity / 1e6),
            # "peak_voltage": self.peak_voltage,
            # "spectro_normalization": self.spectro_normalization,
            # "data_normalization": self.data_normalization,
            # "gain_dB": self.gain_dB
        }
        # TODO: readd `, 'cvr_max':self.dynamic_max, 'cvr_min':self.dynamic_min` above when ok with Aplose
        df = pd.DataFrame.from_records([data])
        df.to_csv(filename, index=False)
        os.chmod(filename, mode=FPDEFAULT)

    # region On cluster

    def preprocess_file(self, audio_file: Path, *, last_file_behavior: Literal["pad","truncate","discard"] = "pad", merge_files: bool = True, write_file: bool = False) -> Tuple[np.ndarray, str]:
        """Preprocess an audio file to prepare the spectrogram creation.
        
        Parameters
        ----------
            audio_file: `pathlib.Path`
                The path to the original audio file
            merge_files: `bool`, optional, keyword-only
                Whether the files should be merged if the durations overlap. If True, will raise an error if the timestamps between two files are not continuous.
                If False, will raise an error if the spectro_duration is greater than the original audio file duration. Default is True.
            write_file: `bool`, optional, keyword-only
                If set to True, will write the preprocessed audio file to the output folder. Else, will hold it in memory. Default is False.
                """
        self.__build_path()

        orig_path = self.path_input_audio_file.joinpath1("old_timestamp.csv") if self.path_input_audio_file.joinpath("old_timestamp.csv").exists() else self.path_input_audio_file.joinpath("timestamp.csv")
        orig_timestamp_file = pd.read_csv(
            orig_path,
            header=None,
            names=["filename", "timestamp", "timezone"],
        )

        final_timestamp_file = pd.read_csv(
            self.audio_path.joinpath("timestamp.csv"),
            header=None,
            names=["filename", "timestamp", "timezone"],
        )

        metadata = pd.read_csv(self.audio_path.joinpath("metadata.csv"))
        #! If spectro == input audio
        if metadata["origin_sr"] == self.dataset_sr and metadata["audio_file_origin_duration"] == self.spectro_duration:
            return (safe_read(audio_file), audio_file)

        files_to_load = [audio_file]
        output_files = []

        orig_index = orig_timestamp_file["filename"] == audio_file
        T0 = orig_timestamp_file[orig_index]["timestamp"].values[0] # Timestamp of the beginning of the original file
        d1 = orig_timestamp_file[orig_index+1]["timestamp"].values[0] - T0 # Duration of the original timestamp (considering all timestamps are continuous)
        final_timestamps = final_timestamp_file["timestamp"].values[0]

        final_timestamps["timestamp"] = [to_timestamp(timestamp) for timestamp in final_timestamps["timestamp"]]

        N0 = final_timestamps[final_timestamp_file["timestamp"] <= T0][-1] # Timestamp of the beginning of the first output file starting before T0
        output_files.append(N0["filename"].values[0])
        N0_index = final_timestamps.index

        start = T0
        i=1
        while start > N0:
            start = orig_timestamp_file[orig_index-i]["timestamp"].values[0]
            files_to_load.insert(0, orig_timestamp_file[orig_index-i]["filename"].values[0])
            i+=1

        offset_beginning = (N0 - start).seconds

        # While the original file ends after the target file, we prepare to create the next.
        j=1
        next_output = N0
        while T0 + d1 > next_output + self.spectro_duration:
            next_output = final_timestamps[N0_index + j]["timestamp"].values[0]
            output_files.append(final_timestamps[N0_index + j]["filename"].values[0])
            j+=1
        
        
        end = T0
        k=1
        while end + d1 < next_output + self.spectro_duration:
            end = orig_timestamp_file[orig_index+k]["timestamp"].values[0]
            files_to_load.append(orig_timestamp_file[orig_index+i]["filename"].values[0])
            k+=1
        
        offset_end = (end + d1 - N0 + self.spectro_duration).seconds
        
        audio_file_origin_duration = metadata["audio_file_origin_duration"][0]

        #! RESHAPING
        # We might reshape the files and create the folder. Note: reshape function might be memory-heavy and deserve a proper qsub job.
        if self.spectro_duration > int(
            audio_file_origin_duration
        ) and not merge_files:
            raise ValueError(
                "Spectrogram size cannot be greater than file duration when not merging files."
            )

        print(
            f"Automatically reshaping audio files to fit the spectro duration value. Files will be {self.spectro_duration} seconds long."
        )

        output = reshape(
            input_files=files_to_load, 
            chunk_size=self.spectro_duration,
            target_sr=self.sr_analysis,
            output_dir_path=self.audio_path,
            offset_beginning=offset_beginning,
            offset_end=offset_end,
            last_file_behavior=last_file_behavior,
            write_output=write_file
            )
    
        return output

    def process_file(
        self, audio_file: Union[str, Path], *, adjust: bool = False, last_file_behavior: Literal["pad","truncate","discard"] = "pad", merge_files: bool = True, write_audio_file: bool = False
    ) -> None:
        """Read an audio file and generate the associated spectrogram.

        Parameters
        ----------
        audio_file : `str` or `Path`
            The name of the audio file to be processed
        adjust : `bool`, optional, keyword-only
            Indicates whether the file should be processed alone to adjust the spectrogram parameters (the default is False)
        save_matrix : `bool`, optional, keyword-only
            Whether to save the spectrogram matrices or not. Note that activating this parameter might increase greatly the volume of the project. (the default is False)
        overwrite: `bool`, optional, keyword-only
            If set to False, will skip the processing if all output files already exist. If set to True, will first delete the existing files before processing.
        clean_adjust_folder: `bool`, optional, keyword-only
            Whether the adjustment folder should be deleted.
        """
        set_umask()

        self.__build_path(self.adjust)
        Zscore = self.zscore_duration if not self.adjust else "original"

        audio_file = Path(audio_file).name

        
        if audio_file not in os.listdir(self.audio_path):
            raise FileNotFoundError(
                f"The file {audio_file} must be in {self.audio_path} in order to be processed."
            )
        
        if self.data_normalization == "zscore" and self.spectro_normalization != "spectrum":
            self.spectro_normalization = "spectrum"
            print("WARNING: the spectrogram normalization has been changed to spectrum because the data will be normalized using zscore.")
        

        print(f"Generating spectrograms for {audio_file}...")
        #! Determination of zscore normalization parameters
        if self.data_normalization == "zscore" and Zscore != "original" and not hasattr(self, "__summStats"):
            average_over_H = int(
                round(pd.to_timedelta(Zscore).total_seconds() / self.spectro_duration)
            )

            df = pd.DataFrame()
            for dd in self.path.joinpath(OSMOSE_PATH.statistics).glob("summaryStats*"):
                df = pd.concat([df, pd.read_csv(dd, header=0)])

            df["mean_avg"] = df["mean"].rolling(average_over_H, min_periods=1).mean()
            df["std_avg"] = df["std"].pow(2).rolling(average_over_H, min_periods=1).mean().apply(np.sqrt, raw=True)

            self.__summStats = df
            self.__zscore_mean = self.__summStats[
            self.__summStats["filename"] == audio_file
            ]["mean_avg"].values[0]
            self.__zscore_std = self.__summStats[
                self.__summStats["filename"] == audio_file
            ]["std_avg"].values[0]

            print(f"Zscore mean : {self.__zscore_mean} and std : {self.__zscore_std}/")

        #! File processing
        audio_data_list = self.preprocess_file(audio_file=audio_file, last_file_behavior=last_file_behavior, merge_files=merge_files, write_file=write_audio_file)

        for data_tuple in audio_data_list:
            data = data_tuple[0]
            outfilename = data_tuple[1]
            output_file = self.path_output_spectrogram.joinpath(outfilename)

            if next(self.path_output_spectrogram.glob(f"{output_file.stem}*"), None) is not None:
                continue

            if self.data_normalization == "instrument":
                data = (
                    (data * self.peak_voltage)
                    / self.sensitivity
                    / 10 ** (self.gain_dB / 20)
                )

            bpcoef = signal.butter(
                20,
                np.array([self.hp_filter_min_freq, self.sr_analysis / 2 - 1]),
                fs=self.sr_analysis,
                output="sos",
                btype="bandpass",
            )
            data = signal.sosfilt(bpcoef, data)

            if adjust:
                make_path(self.path_output_spectrogram, mode=DPDEFAULT)

            if len(audio_data_list) == 1:
                return data
            else:
                yield data


    def gen_spectro(
        self, *, data: np.ndarray, sample_rate: int
    ) -> Tuple[np.ndarray, np.ndarray[float]]:
        """Generate the spectrograms

        Parameters
        ----------
        data : `np.ndarray`
            The audio data from which the tiles will be generated.
        sample_rate : `int`
            The sample rate of the audio data.

        Returns
        -------
        Sxx : `np.NDArray[float64]`
        Freq : `np.NDArray[float]`
        """
        if self.data_normalization == "zscore" and self.zscore_duration:
            if (len(self.zscore_duration) > 0) and (self.zscore_duration != "original"):
                data = (data - self.__zscore_mean) / self.__zscore_std
            elif self.zscore_duration == "original":
                data = (data - np.mean(data)) / np.std(data)

            print(f"data mean : {np.mean(data)} and std : {np.std(data)}")

        Noverlap = int(self.window_size * self.overlap / 100)

        win = np.hamming(self.window_size)
        if self.nfft < (self.window_size):
            if self.spectro_normalization == "density":
                scale_psd = 2.0
            elif self.spectro_normalization == "spectrum":
                scale_psd = 2.0 * sample_rate
        else:
            if self.spectro_normalization == "density":
                scale_psd = 2.0 / (((win * win).sum()) * sample_rate)
            elif self.spectro_normalization == "spectrum":
                scale_psd = 2.0 / ((win * win).sum())

        Nbech = np.size(data)
        Noffset = self.window_size - Noverlap
        Nbwin = int((Nbech - self.window_size) / Noffset)
        Freq = np.fft.rfftfreq(self.nfft, d=1 / sample_rate)

        Sxx = np.zeros([np.size(Freq), Nbwin])
        for idwin in range(Nbwin):
            if self.nfft < (self.window_size):
                x_win = data[idwin * Noffset : idwin * Noffset + self.window_size]
                _, Sxx[:, idwin] = signal.welch(
                    x_win,
                    fs=sample_rate,
                    window="hamming",
                    nperseg=int(self.nfft),
                    noverlap=int(self.nfft / 2),
                    scaling="density",
                )
            else:
                x_win = data[idwin * Noffset : idwin * Noffset + self.window_size] * win
                Sxx[:, idwin] = np.abs(np.fft.rfft(x_win, n=self.nfft)) ** 2
            Sxx[:, idwin] *= scale_psd

        return Sxx, Freq



    # endregion

    def process_all_files(self, *, save_matrix: bool = False):
        """Process all the files in the dataset and generates the spectrograms. It uses the python multiprocessing library
        to parallelise the computation, so it is less efficient to use this method rather than the job scheduler if run on a cluster.
        """

        kwargs = {"save_matrix": save_matrix}

        map_process_file = partial(self.process_file, **kwargs)

        with mp.Pool(processes=min(self.batch_number, mp.cpu_count())) as pool:
            pool.map(map_process_file, self.list_wav_to_process)