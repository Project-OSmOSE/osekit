from functools import partial
import inspect
import os
import sys
from typing import Tuple, Union, Literal
from math import log10
from pathlib import Path
import multiprocessing as mp

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
)  # Not used for now; will be when local execution will be a thing.
from OSmOSE.Dataset import Dataset
from OSmOSE.utils import safe_read
from OSmOSE.config import OSMOSE_PATH


class Spectrogram(Dataset):
    """Main class for spectrogram-related computations. Can resample, reshape and normalize audio files before generating spectrograms."""

    def __init__(
        self,
        dataset_path: str,
        *,
        sr_analysis: int,
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
        sr_analysis : `int`, keyword-only
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
                - HPfilter_min_freq : `int`
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
            analysis_sheet = None
            self.__analysis_file = False
            print(
                "No valid processed/adjust_metadata.csv found and no parameters provided. All attributes will be None."
            )

        self.batch_number: int = batch_number
        self.__sr_analysis: int = sr_analysis

        self.__nfft: int = (
            analysis_sheet["nfft"][0] if analysis_sheet is not None else None
        )
        self.__window_size: int = (
            analysis_sheet["window_size"][0] if analysis_sheet is not None else None
        )
        self.__overlap: int = (
            analysis_sheet["overlap"][0] if analysis_sheet is not None else None
        )
        self.colormap: str = (
            analysis_sheet["colormap"][0] if analysis_sheet is not None else None
        )
        self.__zoom_level: int = (
            analysis_sheet["zoom_level"][0] if analysis_sheet is not None else None
        )
        self.__dynamic_min: int = (
            analysis_sheet["dynamic_min"][0] if analysis_sheet is not None else None
        )
        self.__dynamic_max: int = (
            analysis_sheet["dynamic_max"][0] if analysis_sheet is not None else None
        )
        self.__number_adjustment_spectrogram: int = (
            analysis_sheet["number_adjustment_spectrogram"][0]
            if analysis_sheet is not None
            else None
        )
        self.__spectro_duration: int = (
            analysis_sheet["spectro_duration"][0]
            if analysis_sheet is not None and "spectro_duration" in analysis_sheet
            else -1
        )

        self.__zscore_duration: Union[float, str] = (
            analysis_sheet["zscore_duration"][0]
            if analysis_sheet is not None
            and isinstance(analysis_sheet["zscore_duration"][0], float)
            else None
        )

        # fmin cannot be 0 in butterworth. If that is the case, it takes the smallest value possible, epsilon
        self.__hpfilter_min_freq: int = (
            analysis_sheet["HPfilter_min_freq"][0]
            if analysis_sheet is not None
            and analysis_sheet["HPfilter_min_freq"][0] != 0
            else sys.float_info.epsilon
        )
        sensitivity_dB: int = (
            analysis_sheet["sensitivity_dB"][0] if analysis_sheet is not None else None
        )
        self.__sensitivity: float = (
            10 ** (sensitivity_dB / 20) * 1e6 if analysis_sheet is not None else None
        )
        self.__peak_voltage: float = (
            analysis_sheet["peak_voltage"][0] if analysis_sheet is not None else None
        )
        self.__spectro_normalization: str = (
            analysis_sheet["spectro_normalization"][0]
            if analysis_sheet is not None
            else None
        )
        self.__data_normalization: str = (
            analysis_sheet["data_normalization"][0]
            if analysis_sheet is not None
            else None
        )
        self.__gain_dB: float = (
            analysis_sheet["gain_dB"][0] if analysis_sheet is not None else None
        )

        self.__window_type: str = (
            analysis_sheet["window_type"][0] if analysis_sheet is not None else None
        )

        self.__frequency_resolution: int = (
            analysis_sheet["frequency_resolution"][0]
            if analysis_sheet is not None
            else None
        )

        self.__time_resolution = (
            [
                analysis_sheet[col][0]
                for col in analysis_sheet
                if "time_resolution" in col
            ]
            if analysis_sheet is not None
            else None
        )

        self.Jb = Job_builder()

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

        self.__build_path(transparent=True)

    # region Spectrogram properties

    @property
    def sr_analysis(self):
        """The sampling frequency of the dataset."""
        return self.__sr_analysis

    @sr_analysis.setter
    def sr_analysis(self, value: int):
        self.__sr_analysis = value

    @property
    def nfft(self):
        """The Nonequispaced Fast Fourier Transform of the dataset."""
        return self.__nfft

    @nfft.setter
    def nfft(self, value):
        self.__nfft = value

    @property
    def window_size(self):
        """The window size"""
        return self.__window_size

    @window_size.setter
    def window_size(self, value):
        self.__window_size = value

    @property
    def overlap(self):
        return self.__overlap

    @overlap.setter
    def overlap(self, value):
        self.__overlap = value

    @property
    def colormap(self):
        return self.colormap

    @colormap.setter
    def colormap(self, value):
        self.colormap = value

    @property
    def zoom_level(self):
        return self.__zoom_level

    @zoom_level.setter
    def zoom_level(self, value):
        self.__zoom_level = value

    @property
    def dynamic_min(self):
        return self.__dynamic_min

    @dynamic_min.setter
    def dynamic_min(self, value):
        self.__dynamic_min = value

    @property
    def dynamic_max(self):
        return self.__dynamic_max

    @dynamic_max.setter
    def dynamic_max(self, value):
        self.__dynamic_max = value

    @property
    def number_adjustment_spectrogram(self):
        return self.__number_adjustment_spectrogram

    @number_adjustment_spectrogram.setter
    def number_adjustment_spectrogram(self, value):
        self.__number_adjustment_spectrogram = value

    @property
    def spectro_duration(self):
        return self.__spectro_duration

    @spectro_duration.setter
    def spectro_duration(self, value):
        self.__spectro_duration = value

    @property
    def zscore_duration(self):
        return self.__zscore_duration

    @zscore_duration.setter
    def zscore_duration(self, value):
        self.__zscore_duration = value

    @property
    def HPfilter_min_freq(self):
        return self.__hpfilter_min_freq

    @HPfilter_min_freq.setter
    def HPfilter_min_freq(self, value):
        self.__hpfilter_min_freq = value

    @property
    def sensitivity(self):
        return self.__sensitivity

    @sensitivity.setter
    def sensitivity(self, value):
        """Always assume the sensitivity is given in dB"""
        self.__sensitivity = 10 ** (value / 20) * 1e6

    @property
    def peak_voltage(self):
        return self.__peak_voltage

    @peak_voltage.setter
    def peak_voltage(self, value):
        self.__peak_voltage = value

    @property
    def spectro_normalization(self):
        return self.__spectro_normalization

    @spectro_normalization.setter
    def spectro_normalization(self, value):
        self.__spectro_normalization = value

    @property
    def data_normalization(self):
        return self.__data_normalization

    @data_normalization.setter
    def data_normalization(self, value):
        self.__data_normalization = value

    @property
    def gain_dB(self):
        return self.__gain_dB

    @gain_dB.setter
    def gain_dB(self, value):
        self.__gain_dB = value

    @property
    def window_type(self):
        return self.__window_type

    @window_type.setter
    def window_type(self, value):
        self.__window_type = value

    @property
    def frequency_resolution(self):
        return self.__frequency_resolution

    @frequency_resolution.setter
    def frequency_resolution(self, value):
        self.__frequency_resolution = value

    @property
    def time_resolution(self):
        return self.__time_resolution

    @time_resolution.setter
    def time_resolution(self, value):
        self.__time_resolution = value

    # endregion

    def __build_path(self, adjust: bool = False, transparent: bool = False):
        """Build some internal paths according to the expected architecture. Not path is created.

        Parameter
        ---------
            adjust : `bool`, optional
                Whether or not the paths are used to adjust spectrogram parameters.
            transparent: `bool`, optional
                If set to True, will not create the folders and just return the file path.
        """
        processed_path = self.path.joinpath(OSMOSE_PATH.spectrogram)
        audio_foldername = f"{str(self.spectro_duration)}_{str(self.sr_analysis)}"
        self.audio_path = self.path.joinpath(OSMOSE_PATH.raw_audio, audio_foldername)
        if not transparent:
            self.audio_path.mkdir(mode=0o770, parents=True, exist_ok=True)

        if adjust:
            self.__spectro_foldername = "adjustment_spectros"
        else:
            self.__spectro_foldername = (
                f"{str(self.nfft)}_{str(self.window_size)}_{str(self.overlap)}"
            )

        self.path_output_spectrogram = processed_path.joinpath(
            audio_foldername, self.__spectro_foldername, "image"
        )
        self.path_output_spectrogram.mkdir(mode=0o770, parents=True, exist_ok=True)

        self.__path_summstats = processed_path.joinpath(
            audio_foldername, "normalization_parameters"
        )

        if not transparent:
            self.__path_summstats.mkdir(mode=0o770, parents=True, exist_ok=True)

        self.path_output_spectrogram_matrix = processed_path.joinpath(
            audio_foldername, self.__spectro_foldername, "matrix"
        )

        if not transparent:
            self.path_output_spectrogram_matrix.mkdir(
                mode=0o770, parents=True, exist_ok=True
            )

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

        tile_duration = self.spectro_duration / 2 ** (self.zoom_level - 1)

        data = np.zeros([int(tile_duration * self.sr_analysis), 1])

        Noverlap = int(self.window_size * self.overlap / 100)

        Nbech = np.size(data)
        Noffset = self.window_size - Noverlap
        Nbwin = int((Nbech - self.window_size) / Noffset)
        Freq = np.fft.rfftfreq(self.nfft, d=1 / self.sr_analysis)
        Time = np.linspace(0, Nbech / self.sr_analysis, Nbwin)

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
        sr_analysis: int = None,
        reshape_method: Literal["legacy", "classic", "none"] = "none",
        batch_ind_min: int = 0,
        batch_ind_max: int = -1,
        pad_silence: bool = False,
        force_init: bool = False,
        date_template: str = None,
    ) -> None:
        """Prepares everything (path, variables, files) for spectrogram generation. This needs to be run before the spectrograms are generated.
        If the dataset has not yet been build, it is before the rest of the functions are initialized.

        Parameters
        ----------
        sr_analysis : `int`, optional, keyword-only
            The sampling frequency of the audio files used to generate the spectrograms. If set, will overwrite the Spectrogram.sr_analysis attribute.
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
                    f"Unhandled error during dataset building. They may be resolved by building the dataset separately first. Description of the error: {str(e)}"
                )

        self.__build_path()

        if sr_analysis:
            self.sr_analysis = sr_analysis

        self.path_input_audio_file = self._get_original_after_build()
        list_wav_withEvent_comp = sorted(self.path_input_audio_file.glob("*wav"))

        if batch_ind_max == -1:
            batch_ind_max = len(list_wav_withEvent_comp)
        list_wav_withEvent = list_wav_withEvent_comp[batch_ind_min:batch_ind_max]

        self.list_wav_to_process = [
            audio_file.name for audio_file in list_wav_withEvent
        ]

        # Stop initialization if already done
        final_path = self.path.joinpath(
            OSMOSE_PATH.spectrogram,
            f"{str(self.spectro_duration)}_{str(self.sr_analysis)}",
            f"{str(self.nfft)}_{str(self.window_size)}_{str(self.overlap)}",
            "metadata.csv",
        )
        temp_path = self.path.joinpath(OSMOSE_PATH.spectrogram, "adjust_metadata.csv")
        audio_metadata_path = self.path.joinpath(
            OSMOSE_PATH.raw_audio,
            f"{str(self.spectro_duration)}_{str(self.sr_analysis)}",
            "metadata.csv",
        )

        if (
            (final_path.exists() or temp_path.exists())
            and audio_metadata_path.exists()
            and not force_init
        ):
            audio_file_count = pd.read_csv(audio_metadata_path)["audio_file_count"][0]
            if len(list(audio_metadata_path.parent.glob("*.wav")) == audio_file_count):
                print(
                    "It seems these spectrogram parameters are already initialized. If it is an error or you want to rerun the initialization, add the `force_init` argument."
                )
                return

        # Load variables from raw metadata
        metadata = pd.read_csv(self.path_input_audio_file.joinpath("metadata.csv"))
        audio_file_origin_duration = metadata["audio_file_origin_duration"][0]
        sr_origin = metadata["sr_origin"][0]
        audio_file_count = metadata["audio_file_count"][0]

        if self.path.joinpath(OSMOSE_PATH.processed, "subset_files.csv").is_file():
            subset = pd.read_csv(
                self.path.joinpath(OSMOSE_PATH.processed, "subset_files.csv"),
                header=None,
            )[0].values
            self.list_wav_to_process = list(
                set(subset).intersection(set(self.list_wav_to_process))
            )

        batch_size = len(self.list_wav_to_process) // self.batch_number

        #! RESAMPLING
        resample_job_id_list = []
        processes = []

        if self.sr_analysis != sr_origin and not os.listdir(self.audio_path):
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
                            "target_sr": self.sr_analysis,
                            "batch_ind_min": i_min,
                            "batch_ind_max": i_max,
                        },
                    )

                    process.start()
                    processes.append(process)
                else:
                    jobfile = self.Jb.build_job_file(
                        script_path=Path(inspect.getfile(resample)).resolve(),
                        script_args=f"--input-dir {self.path_input_audio_file} --target-sr {self.sr_analysis} --ind-min {i_min} --ind-max {i_max} --output-dir {self.audio_path}",
                        jobname="OSmOSE_resample",
                        preset="low",
                    )
                    # TODO: use importlib.resources

                    job_id = self.Jb.submit_job(jobfile)
                    resample_job_id_list.append(job_id)

            for process in processes:
                process.join()

        #! ZSCORE NORMALIZATION
        isnorma = (
            any([cc in self.zscore_duration for cc in ["D", "M", "H", "S", "W"]])
            if self.zscore_duration
            else False
        )

        norma_job_id_list = []
        if (
            os.listdir(self.__path_summstats)
            and self.data_normalization == "zscore"
            and isnorma
        ):
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
                            "output_file": self.__path_summstats.joinpath(
                                "SummaryStats_" + str(i_min) + ".csv"
                            ),
                            "target_sr": self.sr_analysis,
                            "batch_ind_min": i_min,
                            "batch_ind_max": i_max,
                        },
                    )

                    process.start()
                    processes.append(process)
                else:
                    jobfile = self.Jb.build_job_file(
                        script_path=Path(inspect.getfile(compute_stats)).resolve(),
                        script_args=f"--input-dir {self.path_input_audio_file} --hpfilter-min-freq {self.HPfilter_min_freq} \
                                    --ind-min {i_min} --ind-max {i_max} --output-file {self.__path_summstats.joinpath('SummaryStats_' + str(i_min) + '.csv')}",
                        jobname="OSmOSE_get_zscore_params",
                        preset="low",
                    )

                    job_id = self.Jb.submit_job(
                        jobfile, dependency=resample_job_id_list
                    )
                    norma_job_id_list.append(job_id)

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
                                "input_files": self.path_input_audio_file,
                                "chunk_size": self.spectro_duration,
                                "output_dir_path": self.audio_path,
                                "offset_beginning": int(offset_beginning),
                                "offset_end": int(offset_end),
                                "batch_ind_min": i_min,
                                "batch_ind_max": i_max,
                                "last_file_behavior": last_file_behavior,
                            },
                        )

                        process.start()
                        processes.append(process)
                    else:
                        jobfile = self.Jb.build_job_file(
                            script_path=Path(inspect.getfile(reshape)).resolve(),
                            script_args=f"--input-files {self.path_input_audio_file} --chunk-size {self.spectro_duration} --ind-min {i_min}\
                                        --ind-max {i_max} --output-dir {self.audio_path} --offset-beginning {int(offset_beginning)} --offset-end {int(offset_end)}\
                                        --last-file-behavior {last_file_behavior} {'--force' if force_init else ''}",
                            jobname="OSmOSE_reshape_py",
                            preset="low",
                        )

                        job_id = self.Jb.submit_job(
                            jobfile, dependency=norma_job_id_list
                        )
                        reshape_job_id_list.append(job_id)

                for process in processes:
                    process.join()

            elif reshape_method == "legacy":
                silence_arg = "-s" if pad_silence else ""
                for batch in range(self.batch_number):
                    i_min = batch * batch_size
                    i_max = (
                        i_min + batch_size
                        if batch < self.batch_number - 1
                        else len(self.list_wav_to_process)
                    )  # If it is the last batch, take all files
                    jobfile = self.Jb.build_job_file(
                        script_path=Path(__file__.parent, "cluster", "reshaper.sh"),
                        script_args=f"-d {self.path} -i {self.path_input_audio_file.name} -t {sr_analysis} \
                                    -m {i_min} -x {i_max} -o {self.audio_path} -n {self.spectro_duration} {silence_arg}",
                        jobname="OSmOSE_reshape_bash",
                        preset="low",
                    )

                    job_id = self.Jb.submit_job(
                        jobfile, dependency=resample_job_id_list
                    )
                    reshape_job_id_list.append(job_id)

        metadata["dataset_fileDuration"] = self.spectro_duration
        new_meta_path = self.audio_path.joinpath("metadata.csv")
        metadata.to_csv(new_meta_path)

        for path in [
            self.path_output_spectrogram,
            self.path_output_spectrogram_matrix,
        ]:
            path.mkdir(mode=0o770, parents=True, exist_ok=True)

        # self.to_csv(os.path.join(self.path_output_spectrograms, "spectrograms.csv"))

        if not self.__analysis_file:
            data = {
                "dataset_name": self.name,
                "sr_analysis": float(self.sr_analysis),
                "nfft": self.nfft,
                "window_size": self.window_size,
                "overlap": self.overlap,
                "colormap": self.colormap,
                "zoom_level": self.zoom_level,
                "number_adjustment_spectrogram": self.number_adjustment_spectrogram,
                "dynamic_min": self.dynamic_min,
                "dynamic_max": self.dynamic_max,
                "spectro_duration": self.spectro_duration,
                "folderName_audioFiles": self.audio_path.name,
                "data_normalization": self.data_normalization,
                "HPfilter_min_freq": self.HPfilter_min_freq,
                "sensitivity_dB": 20 * log10(self.sensitivity / 1e6),
                "peak_voltage": self.peak_voltage,
                "spectro_normalization": self.spectro_normalization,
                "gain_dB": self.gain_dB,
                "zscore_duration": self.zscore_duration,
            }
            analysis_sheet = pd.DataFrame.from_records([data])
            analysis_sheet.to_csv(
                self.path.joinpath(OSMOSE_PATH.spectrogram, "adjust_metadata.csv")
            )

    def to_csv(self, filename: str) -> None:
        """Outputs the characteristics of the spectrogram the specified file in csv format.

        Parameter
        ---------
        filename: str
            The name of the file to be written."""

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
            # "HPfilter_min_freq": self.HPfilter_min_freq,
            # "sensitivity_dB": 20 * log10(self.sensitivity / 1e6),
            # "peak_voltage": self.peak_voltage,
            # "spectro_normalization": self.spectro_normalization,
            # "data_normalization": self.data_normalization,
            # "gain_dB": self.gain_dB
        }
        # TODO: readd `, 'cvr_max':self.dynamic_max, 'cvr_min':self.dynamic_min` above when ok with Aplose
        df = pd.DataFrame.from_records([data])
        df.to_csv(filename, index=False)

    # region On cluster

    def process_file(
        self, audio_file: str, *, adjust: bool = False, save_matrix: bool = False
    ) -> None:
        """Read an audio file and generate the associated spectrogram.

        Parameters
        ----------
        audio_file : `str`
            The name of the audio file to be processed
        adjust : `bool`, optional, keyword-only
            Indicates whether the file should be processed alone to adjust the spectrogram parameters (the default is False)
        save_matrix : `save_matrix`, optional, keyword-only
            Whether to save the spectrogram matrices or not. Note that activating this parameter might increase greatly the volume of the project. (the default is False)
        """
        self.__build_path(adjust)
        self.save_matrix = save_matrix
        self.adjust = adjust
        Zscore = self.zscore_duration if not adjust else "original"

        #! Determination of zscore normalization parameters
        if Zscore and self.data_normalization == "zscore" and Zscore != "original":
            average_over_H = int(
                round(pd.to_timedelta(Zscore).total_seconds() / self.spectro_duration)
            )

            df = pd.DataFrame()
            for dd in self.__path_summstats.glob("summaryStats*"):
                df = pd.concat([df, pd.read_csv(dd, header=0)])

            df["mean_avg"] = df["mean"].rolling(average_over_H, min_periods=1).mean()
            df["std_avg"] = df["std"].rolling(average_over_H, min_periods=1).std()

            self.__summStats = df

        audio_file = Path(audio_file).name
        if audio_file not in os.listdir(self.audio_path):
            raise FileNotFoundError(
                f"The file {audio_file} must be in {self.audio_path} in order to be processed."
            )

        if Zscore and Zscore != "original" and self.data_normalization == "zscore":
            self.__zscore_mean = self.__summStats[
                self.__summStats["filename"] == audio_file
            ]["mean_avg"].values[0]
            self.__zscore_std = self.__summStats[
                self.__summStats["filename"] == audio_file
            ]["std_avg"].values[0]

        #! File processing
        data, sample_rate = safe_read(self.audio_path.joinpath(audio_file))

        if self.data_normalization == "instrument":
            data = (
                (data * self.peak_voltage)
                / self.sensitivity
                / 10 ** (self.gain_dB / 20)
            )

        bpcoef = signal.butter(
            20,
            np.array([self.HPfilter_min_freq, sample_rate / 2 - 1]),
            fs=sample_rate,
            output="sos",
            btype="bandpass",
        )
        data = signal.sosfilt(bpcoef, data)

        if adjust:
            self.path_output_spectrogram.mkdir(mode=0o770, parents=True, exist_ok=True)

        output_file = self.path_output_spectrogram.joinpath(audio_file)

        self.gen_tiles(data=data, sample_rate=sample_rate, output_file=output_file)

    def gen_tiles(self, *, data: np.ndarray, sample_rate: int, output_file: Path):
        """Generate spectrogram tiles corresponding to the zoom levels.

        Parameters
        ----------
        data : `np.ndarray`
            The audio data from which the tiles will be generated.
        sample_rate : `int`
            The sample rate of the audio data.
        output_file : `str`
            The name of the output spectrogram."""
        if self.data_normalization == "zscore" and self.zscore_duration:
            if (len(self.zscore_duration) > 0) and (self.zscore_duration != "original"):
                data = (data - self.__zscore_mean) / self.__zscore_std
            elif self.zscore_duration == "original":
                data = (data - np.mean(data)) / np.std(data)

        duration = len(data) / int(sample_rate)

        nber_tiles_lowest_zoom_level = 2 ** (self.zoom_level - 1)
        tile_duration = duration / nber_tiles_lowest_zoom_level

        Sxx_2 = np.empty((int(self.nfft / 2) + 1, 1))
        for tile in range(0, nber_tiles_lowest_zoom_level):
            start = tile * tile_duration
            end = start + tile_duration

            sample_data = data[int(start * sample_rate) : int((end + 1) * sample_rate)]

            output_file = output_file.parent.joinpath(
                f"{output_file.stem}_{str(nber_tiles_lowest_zoom_level)}_{str(tile)}.png"
            )

        Sxx, Freq = self.gen_spectro(
            data=sample_data, sample_rate=sample_rate, output_file=output_file
        )

        Sxx_2 = np.hstack((Sxx_2, Sxx))

        Sxx_lowest_level = Sxx_2[:, 1:]

        segment_times = np.linspace(
            0, len(data) / sample_rate, Sxx_lowest_level.shape[1]
        )[np.newaxis, :]

        # loop over the zoom levels from the second lowest to the highest one
        for zoom_level in range(self.zoom_level)[::-1]:
            nberspec = Sxx_lowest_level.shape[1] // (2**zoom_level)

            # loop over the tiles at each zoom level
            for tile in range(2**zoom_level):
                Sxx_int = Sxx_lowest_level[:, tile * nberspec : (tile + 1) * nberspec][
                    :, :: 2 ** (self.zoom_level - zoom_level)
                ]

                segment_times_int = segment_times[
                    :, tile * nberspec : (tile + 1) * nberspec
                ][:, :: 2 ** (self.zoom_level - zoom_level)]

                if self.spectro_normalization == "density":
                    log_spectro = 10 * np.log10(Sxx_int / (1e-12))
                if self.spectro_normalization == "spectrum":
                    log_spectro = 10 * np.log10(Sxx_int)

                self.generate_and_save_figures(
                    time=segment_times_int,
                    freq=Freq,
                    log_spectro=log_spectro,
                    output_file=output_file.parent.joinpath(
                        f"{output_file.stem}_{str(2 ** zoom_level)}_{str(tile)}.png"
                    ),
                )

    def gen_spectro(
        self, *, data: np.ndarray, sample_rate: int, output_file: Path
    ) -> Tuple[np.ndarray, np.ndarray[float]]:
        """Generate the spectrograms

        Parameters
        ----------
        data : `np.ndarray`
            The audio data from which the tiles will be generated.
        sample_rate : `int`
            The sample rate of the audio data.
        output_file : `str`
            The name of the output spectrogram.

        Returns
        -------
        Sxx : `np.NDArray[float64]`
        Freq : `np.NDArray[float]`
        """
        Noverlap = int(self.window_size * self.overlap / 100)

        win = np.hamming(self.window_size)
        if self.nfft < (0.5 * self.window_size):
            if self.spectro_normalization == "density":
                scale_psd = 2.0
            if self.spectro_normalization == "spectrum":
                scale_psd = 2.0 * sample_rate
        else:
            if self.spectro_normalization == "density":
                scale_psd = 2.0 / (((win * win).sum()) * sample_rate)
            if self.spectro_normalization == "spectrum":
                scale_psd = 2.0 / ((win * win).sum())

        Nbech = np.size(data)
        Noffset = self.window_size - Noverlap
        Nbwin = int((Nbech - self.window_size) / Noffset)
        Freq = np.fft.rfftfreq(self.nfft, d=1 / sample_rate)

        Sxx = np.zeros([np.size(Freq), Nbwin])
        Time = np.linspace(0, Nbech / sample_rate, Nbwin)
        for idwin in range(Nbwin):
            if self.nfft < (0.5 * self.window_size):
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

        if self.spectro_normalization == "density":
            log_spectro = 10 * np.log10((Sxx / (1e-12)) + (1e-20))
        if self.spectro_normalization == "spectrum":
            log_spectro = 10 * np.log10(Sxx + (1e-20))

        # save spectrogram as a png image
        self.generate_and_save_figures(
            time=Time, freq=Freq, log_spectro=log_spectro, output_file=output_file
        )

        # save spectrogram matrices (intensity, time and freq) in a npz file
        if self.save_matrix:
            self.path_output_spectrogram_matrix.mkdir(
                mode=0o770, parents=True, exist_ok=True
            )
            output_matrix = self.path_output_spectrogram_matrix.joinpath(
                output_file.name
            ).with_suffix(".npz")

            np.savez(
                output_matrix,
                Sxx=Sxx,
                log_spectro=log_spectro,
                Freq=Freq,
                Time=Time,
            )

        return Sxx, Freq

    def generate_and_save_figures(
        self,
        *,
        time: np.ndarray[float],
        freq: np.ndarray[float],
        log_spectro: np.ndarray[int],
        output_file: Path,
    ):
        """Write the spectrogram figures to the output file.

        Parameters
        ----------
        time : `np.NDArray[floating]`
        freq : `np.NDArray[floating]`
        log_spectro : `np.NDArray[signed int]`
        output_file : `str`
            The name of the spectrogram file."""
        # Plotting spectrogram
        my_dpi = 100
        fact_x = 1.3
        fact_y = 1.3
        fig, ax = plt.subplots(
            nrows=1,
            ncols=1,
            figsize=(fact_x * 1800 / my_dpi, fact_y * 512 / my_dpi),
            dpi=my_dpi,
        )
        color_map = plt.cm.get_cmap(self.colormap)  # .reversed()
        plt.pcolormesh(time, freq, log_spectro, cmap=color_map)
        plt.clim(vmin=self.dynamic_min, vmax=self.dynamic_max)
        # plt.colorbar()

        # If generate all
        fig.axes[0].get_xaxis().set_visible(True)
        fig.axes[0].get_yaxis().set_visible(True)
        ax.set_frame_on(True)

        ax.spines["right"].set_visible(True)
        ax.spines["left"].set_visible(True)
        ax.spines["bottom"].set_visible(True)
        ax.spines["top"].set_visible(True)

        # For test
        fig.axes[0].get_xaxis().set_visible(True)
        fig.axes[0].get_yaxis().set_visible(True)
        ax.set_ylabel("Frequency (Hz)")
        ax.set_xlabel("Time (s)")
        plt.colorbar()

        # Saving spectrogram plot to file
        plt.savefig(output_file, bbox_inches="tight", pad_inches=0)
        plt.close()

        metadata_input = self.path.joinpath(
            OSMOSE_PATH.spectrogram, "adjust_metadata.csv"
        )
        metadata_output = self.path.joinpath(
            OSMOSE_PATH.spectrogram,
            f"{str(self.spectro_duration)}_{str(self.sr_analysis)}",
            f"{str(self.nfft)}_{str(self.window_size)}_{str(self.overlap)}",
            "metadata.csv",
        )
        print()
        if not self.adjust and metadata_input.exists() and not metadata_output.exists():
            metadata_input.rename(metadata_output)

    # endregion

    def process_all_files(self, *, save_matrix: bool = False):
        """Process all the files in the dataset and generates the spectrograms. It uses the python multiprocessing library
        to parallelise the computation, so it is less efficient to use this method rather than the job scheduler if run on a cluster.
        """

        kwargs = {"save_matrix": save_matrix}

        map_process_file = partial(self.process_file, **kwargs)

        with mp.Pool(processes=min(self.batch_number, mp.cpu_count())) as pool:
            pool.map(map_process_file, self.list_wav_to_process)
