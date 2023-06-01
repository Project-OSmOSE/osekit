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

from OSmOSE.features import compute_statistics
from OSmOSE.core import Dataset, Job_builder
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
            - Use the existing `analysis/self.__analysis_sheet.csv` file. If one exist, it will take priority over the other methods. Note that
            when using this file, some attributes will be locked in read-only mode.
            - Fill the `analysis_params` argument. More info on the expected value below.
            - Don't initialize the attributes in the constructor, and assign their values manually.

        In any case, all attributes must have a value for the spectrograms to be generated. If it does not exist, `analysis/self.__analysis_sheet.csv`
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
            If `analysis/self.__analysis_sheet.csv` does not exist, the analysis parameters can be submitted in the form of a dict,
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
            If additional information is given, it will be ignored. Note that if there is an `analysis/self.__analysis_sheet.csv` file, it will
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
            self.__analysis_sheet = pd.read_csv(metadata_path, header=0)
        elif analysis_params:
            # We put the value in a list so that value[0] returns the right value below.
            self.__analysis_sheet = {key: [value] for (key, value) in analysis_params.items()}
        else:
            self.__analysis_sheet = {}
            print(
                "No valid processed/adjust_metadata.csv found and no parameters provided. All attributes will be initialized to default values..  \n"
            )

        self.batch_number: int = batch_number
        self.dataset_sr: int = dataset_sr if dataset_sr is not None else orig_metadata['origin_sr'][0]

        self.nfft: int = self.__analysis_sheet["nfft"][0] if "nfft" in self.__analysis_sheet else 1
        self.window_size: int = (
            self.__analysis_sheet["window_size"][0]
            if "window_size" in self.__analysis_sheet
            else None
        )
        self.overlap: int = (
            self.__analysis_sheet["overlap"][0] if "overlap" in self.__analysis_sheet else None
        )
        self.number_adjustment_spectrogram: int = (
            self.__analysis_sheet["number_adjustment_spectrogram"][0]
            if "number_adjustment_spectrogram" in self.__analysis_sheet
            else None
        )
        self.spectro_duration: int = (
            self.__analysis_sheet["spectro_duration"][0]
            if self.__analysis_sheet is not None and "spectro_duration" in self.__analysis_sheet
            else (
                orig_metadata["audio_file_origin_duration"][0]
                if self.is_built
                else -1
            )
        )

        self.zscore_duration: Union[float, str] = (
            self.__analysis_sheet["zscore_duration"][0]
            if "zscore_duration" in self.__analysis_sheet
            and isinstance(self.__analysis_sheet["zscore_duration"][0], float)
            else "original"
        )

        # fmin cannot be 0 in butterworth. If that is the case, it takes the smallest value possible, epsilon
        self.hp_filter_min_freq: int = (
            self.__analysis_sheet["hp_filter_min_freq"][0]
            if "hp_filter_min_freq" in self.__analysis_sheet
            else 0
        )

        self.sensitivity: float = (
            self.__analysis_sheet["sensitivity_dB"][0]
            if "sensitivity_dB" in self.__analysis_sheet
            else 0
        )

        self.peak_voltage: float = (
            self.__analysis_sheet["peak_voltage"][0]
            if "peak_voltage" in self.__analysis_sheet
            else None
        )
        self.spectro_normalization: str = (
            self.__analysis_sheet["spectro_normalization"][0]
            if "spectro_normalization" in self.__analysis_sheet
            else None
        )
        self.data_normalization: str = (
            self.__analysis_sheet["data_normalization"][0]
            if "data_normalization" in self.__analysis_sheet
            else None
        )
        self.gain_dB: float = (
            self.__analysis_sheet["gain_dB"][0]
            if "gain_dB" in self.__analysis_sheet is not None
            else None
        )

        self.window_type: str = (
            self.__analysis_sheet["window_type"][0]
            if "window_type" in self.__analysis_sheet
            else "hamming"
        )

        self.time_resolution = (
            [
                self.__analysis_sheet[col][0]
                for col in self.__analysis_sheet
                if "time_resolution" in col
            ]
            if self.__analysis_sheet is not None
            else [0] * self.zoom_level
        )

        self.jb = Job_builder()


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

    # TODO: some cleaning
    def initialize(self, *, date_template:str = None, force_init: bool = False, batch_ind_min:int = 0, batch_ind_max:int = -1):
        audio_foldername = f"{str(self.spectro_duration)}_{str(self.dataset_sr)}"
        self.audio_path = self.path.joinpath(OSMOSE_PATH.raw_audio, audio_foldername)
        # Mandatory init
        if not self.is_built:
            try:
                self.build(date_template=date_template)
            except Exception as e:
                print(
                    f"Unhandled error during dataset building. The spectrogram initialization will be cancelled. The error may be resolved by building the dataset separately first. Description of the error: {str(e)}"
                )
                return
            
        if self.data_normalization == "zscore" and self.spectro_normalization != "spectrum":
            self.spectro_normalization = "spectrum"
            print("WARNING: the spectrogram normalization has been changed to spectrum because the data will be normalized using zscore.")

        self.path_input_audio_file = self._get_original_after_build()
        
        #! INITIALIZATION START
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
        processes = []
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
                            "hp_filter_min_freq": self.hp_filter_min_freq,
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

                    job_id = self.jb.submit_job()
                    norma_job_id_list += job_id
            
            self.pending_jobs = norma_job_id_list

            for process in processes:
                process.join()


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
        
    # region On cluster

    def preprocess_file(self, audio_file: Path, *, last_file_behavior: Literal["pad","truncate","discard"] = "pad", merge_files: bool = True, write_file: bool = False, adjust: bool = False) -> Tuple[np.ndarray, str]:
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

        #! Data validation
        if audio_file not in os.listdir(self.audio_path):
            raise FileNotFoundError(
                f"The file {audio_file} must be in {self.audio_path} in order to be processed."
            )
        
        if self.data_normalization == "zscore" and self.spectro_normalization != "spectrum":
            self.spectro_normalization = "spectrum"
            print("WARNING: the spectrogram normalization has been changed to spectrum because the data will be normalized using zscore.")
        


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



        orig_path = self.path_input_audio_file.joinpath("orig_timestamp.csv") if self.path_input_audio_file.joinpath("orig_timestamp.csv").exists() else self.path_input_audio_file.joinpath("timestamp.csv")
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

        reshaped = reshape(
            input_files=files_to_load, 
            chunk_size=self.spectro_duration,
            target_sr=self.sr_analysis,
            output_dir_path=self.audio_path,
            offset_beginning=offset_beginning,
            offset_end=offset_end,
            last_file_behavior=last_file_behavior,
            write_output=write_file
            )
    
        for data_tuple in reshaped:
            data = data_tuple[0]
            outfilename = data_tuple[1]
            output_file = self.path_output_spectrogram.joinpath(outfilename)

            if next(self.path_output_spectrogram.glob(f"{output_file.stem}*"), None) is not None:
                continue

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

            yield data, output_file


    def compute_welch(
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

        if self.data_normalization == "instrument":
            data = (
                (data * self.peak_voltage)
                / self.sensitivity
                / 10 ** (self.gain_dB / 20)
            )

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
