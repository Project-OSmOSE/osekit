import os
import shutil
from datetime import timedelta
from filelock import FileLock
from typing import Literal, Union
from termcolor import colored
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from OSmOSE.utils.timestamp_utils import to_timestamp
from OSmOSE.config import *
from OSmOSE.features import Welch
from OSmOSE.utils import set_umask, make_path

class Aplose(Welch):
    def __init__(self, dataset_path: str, *, dataset_sr: int = 0, gps_coordinates: str | list | tuple = None, owner_group: str = None, analysis_params: dict = None, batch_number: int = 10, local: bool = False) -> None:
        super().__init__(dataset_path, dataset_sr=dataset_sr, gps_coordinates=gps_coordinates, owner_group=owner_group, analysis_params=analysis_params, batch_number=batch_number, local=local)

        self.colormap: str = (
            self.analysis_sheet["colormap"][0] if "colormap" in self.analysis_sheet else "viridis"
        )
        self.zoom_level: int = (
            self.analysis_sheet["zoom_level"][0] if "zoom_level" in self.analysis_sheet else 0
        )
        self.dynamic_min: int = (
            self.analysis_sheet["dynamic_min"][0]
            if "dynamic_min" in self.analysis_sheet
            else None
        )
        self.dynamic_max: int = (
            self.analysis_sheet["dynamic_max"][0]
            if "dynamic_max" in self.analysis_sheet
            else None
        )
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
        self.build_path(dry=True)

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


    def build_path(self, adjust: bool = False, dry: bool = False):
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

    def initialize(
        self,
        *,
        dataset_sr: int = None,
        force_init: bool = False,
        date_template: str = None
    ) -> None:
        """Prepares everything (path, variables, files) for spectrogram generation. This needs to be run before the spectrograms are generated.
        If the dataset has not yet been build, it is before the rest of the functions are initialized.

        Parameters
        ----------
        dataset_sr : `int`, optional, keyword-only
            The sampling frequency of the audio files used to generate the spectrograms. If set, will overwrite the Spectrogram.dataset_sr attribute.
        force_init : `bool`, optional, keyword-only
            Force every parameter of the initialization.
        date_template : `str`, optiona, keyword-only
            When initializing a spectrogram of a dataset that has not been built, providing a date_template will generate the timestamp.csv.
        """
        self.build_path()
        super().initialize(force_init=force_init, date_template=date_template)

        if dataset_sr:
            self.dataset_sr = dataset_sr


    def generate_spectrogram(
        self, 
        audio_file: Union[str, Path], 
        *, 
        adjust: bool = False, 
        save_matrix: bool = False, 
        save_image: bool = False,
        last_file_behavior: Literal["pad","truncate","discard"] = "pad", 
        merge_files: bool = True, 
        write_audio_file: bool = False,
        clean_adjust_folder: bool = False,
        overwrite: bool = False
    ) -> None:
        """Generate all spectrograms of a given file. 
        
        If the file overlaps several spectrograms in the timeline, they all will be generated.
        
        Parameter
        ---------
            audio_file: str or Path
                The path to the audio file to be processed.
            adjust: bool, optional, keyword-only
                Whether or not the spectrogram is generated to adjust parameters. Default is False.
            save_matrix: bool, optional, keyword-only
                Save the spectrogram's matrix. At least one of this or save_image MUST be set to True. Default is False.
            save_image: bool, optional, keyword-only
                Save the spectrogram's image. Default is False.
            last_file_behavior:  `{"truncate","pad","discard"}, optional, keyword-only
                When reshaping multiple files, what to do with if the last data of the last file is too small to fill a whole file.
                    - `truncate` creates a truncated file with the remaining data, which will have a different duration than the others.
                    - `pad` creates a file of the same duration than the others, where the missing data is filled with 0.
                    - `discard` ignores the remaining data. The last seconds/minutes/hours of audio will be lost in the reshaping.
                The default method is `pad`.
            merge_files: bool, optional, keyword-only
                Merge continuous files during reshaping. Will only merge them if there is less than 5 secondes between the two files. Default is True.
            write_audio_file: bool, optional, keyword-only
                Wite the reshaped and resampled audio file(s). Default is False.
            clean_adjust_folder: bool, optional, keyword-only
                Clean the adjustment folder before generating the spectrogram. Useful to not clutter the folder. Default is False.
            overwrite: bool, optional, keyword-only
                Force to rewrite every file even if it already exists. Will not delete files that would not be written. Default is False.
                """

        if not save_image and not save_matrix:
            raise ValueError("Neither image or matrix are set to be generated. Please set at least one of save_matrix or save_image to True to proceed with the spectrogram generation, or use the welch() method to get the raw data.")

        self.build_path(adjust=adjust)

        set_umask()
        try:
            if clean_adjust_folder and (self.path_output_spectrogram.parent.parent.joinpath(
                    "adjustment_spectros"
                ).exists()
            ):
                shutil.rmtree(
                    self.path_output_spectrogram.parent.parent.joinpath(
                        "adjustment_spectros"
                    ), ignore_errors=True
                )
                if adjust:
                    make_path(self.path_output_spectrogram.parent.parent.joinpath(
                        "adjustment_spectros"
                    ), mode=DPDEFAULT)
                    make_path(self.path_output_spectrogram, mode=DPDEFAULT)
        except: 
            pass

        self.save_matrix = save_matrix
        self.save_image = save_image
        
        self.adjust = adjust
        lockfile = f"{self.path_output_spectrogram.joinpath('lock' + Path(audio_file).stem)}.lock"

        def check_existing_matrix():
            return len(list(self.path_output_spectrogram_matrix.glob(f"{Path(audio_file).stem}*"))) == 2**self.zoom_level if save_matrix else True

        if len(list(self.path_output_spectrogram.glob(f"{Path(audio_file).stem}*"))) == sum(2**i for i in range(self.zoom_level + 1)) and check_existing_matrix():
            if overwrite:
                print(f"Existing files detected for audio file {audio_file}! They will be overwritten.")
                for old_file in self.path_output_spectrogram.glob(f"{Path(audio_file).stem}*"):
                    old_file.unlink()
                if save_matrix:
                    for old_matrix in self.path_output_spectrogram_matrix.glob(f"{Path(audio_file).stem}*"):
                        old_matrix.unlink()
            else:
                print(f"The spectrograms for the file {audio_file} have already been generated, skipping...")
                return

        lock = FileLock(lockfile)
        lock.acquire(blocking=False)

        welchs = self.preprocess_file(audio_file=audio_file,
                          adjust=adjust,
                          last_file_behavior=last_file_behavior,
                          merge_files=merge_files,
                          write_file=write_audio_file)

        for welch in welchs:
            self.gen_tiles(data=welch[0], sample_rate=self.dataset_sr, output_file=self.path_output_spectrogram.joinpath(welch[1]))

        lock.release()
        try:
            os.remove(lockfile)
        except:
            pass


    def gen_tiles(self, *, data: np.ndarray, sample_rate: int, output_file: Path):
        """Generate spectrogram tiles corresponding to the zoom levels.

        Parameters
        ----------
        data : `np.ndarray`
            The audio data from which the tiles will be generated.
        sample_rate : `int`
            The sample rate of the audio data.
        output_file : `Path`
            The name of the output spectrogram."""

        duration = len(data) / int(sample_rate)

        nber_tiles_lowest_zoom_level = 2 ** (self.zoom_level)
        tile_duration = duration / nber_tiles_lowest_zoom_level
        
        timestamp_list = []
        # Try to retrieve timestamp from the filename, and if fails fallback to looking in the csv
        try:
            current_timestamp = to_timestamp(output_file.stem)
        except ValueError:
            timestamp_file = pd.read_csv(self.audio_path.joinpath("timestamp.csv"), header=None, names=["filename","timestamp","timezone"])
            current_timestamp = to_timestamp(timestamp_file["timestamp"][timestamp_file["filename"] == f"{output_file.stem}.wav"].values[0])


        Sxx_complete_lowest_level = np.empty((int(self.nfft / 2) + 1, 1))
        Sxx_mean_lowest_tuile = np.empty((1,int(self.nfft / 2) + 1))
        for tile in range(0, nber_tiles_lowest_zoom_level):
            start = tile * tile_duration
            end = start + tile_duration

            timestamp_list.append(current_timestamp + timedelta(seconds=int(start)))
            sample_data = data[int(start * sample_rate) : int(end * sample_rate) -1]

            Sxx, Freq = self.compute_welch(
                data=sample_data,
                sample_rate=sample_rate,
            )

            Sxx_complete_lowest_level = np.hstack((Sxx_complete_lowest_level, Sxx))
            Sxx_mean_lowest_tuile = np.vstack((Sxx_mean_lowest_tuile, Sxx.mean(axis=1)[np.newaxis,:]))

            # save spectrogram matrices (intensity, time and freq) in a npz file
            if self.save_matrix:
                Nbech = np.size(data)
                Noffset = self.window_size - int(self.window_size * self.overlap / 100)
                Nbwin = int((Nbech - self.window_size) / Noffset)
                Time = np.linspace(0, Nbech / sample_rate, Nbwin)
                if self.spectro_normalization == "density":
                    log_spectro = 10 * np.log10((Sxx / (1e-12)) + (1e-20))
                if self.spectro_normalization == "spectrum":
                    log_spectro = 10 * np.log10(Sxx + (1e-20))

                make_path(self.path_output_spectrogram_matrix, mode=DPDEFAULT)

                output_matrix = self.path_output_spectrogram_matrix.joinpath(
                    output_file.name
                ).with_suffix(".npz")

                if not output_matrix.exists():
                    np.savez(
                        output_matrix,
                        Sxx=Sxx,
                        log_spectro=log_spectro,
                        Freq=Freq,
                        Time=Time,
                    )

                    os.chmod(output_matrix, mode=FPDEFAULT)

                os.chmod(output_matrix, mode=FPDEFAULT)

        Sxx_complete_lowest_level = Sxx_complete_lowest_level[:, 1:]
        Sxx_mean_lowest_tuile = Sxx_mean_lowest_tuile[1:,:]

        segment_times = np.linspace(
            0, data.size / sample_rate, Sxx_complete_lowest_level.shape[1]
        )[np.newaxis, :]

        #time resolution not yet implemented
        #self.time_resolution = [segment_times[1] - segment_times[0]]

        if self.save_image:
            # loop over the zoom levels from the second lowest to the highest one
            for zoom_level in range(self.zoom_level + 1)[::-1]:
                nberspec = Sxx_complete_lowest_level.shape[1] // (2**zoom_level)

                # loop over the tiles at each zoom level
                for tile in range(2**zoom_level):
                    Sxx_int = Sxx_complete_lowest_level[:, tile * nberspec : (tile + 1) * nberspec][
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
                        output_file=self.path_output_spectrogram.joinpath(
                            f"{output_file.stem}_{str(2 ** zoom_level)}_{str(tile)}.png"
                        ),
                        adjust=self.adjust
                    )

    def generate_and_save_figures(
        self,
        *,
        time: np.ndarray[float],
        freq: np.ndarray[float],
        log_spectro: np.ndarray[int],
        output_file: Path,
        adjust: bool
    ):
        """Write the spectrogram figures to the output file.

        Parameters
        ----------
        time : `np.NDArray[floating]`
        freq : `np.NDArray[floating]`
        log_spectro : `np.NDArray[signed int]`
        output_file : `str`
            The name of the spectrogram file."""
        if output_file.exists(): 
            print(f"The spectrogram {output_file.name} has already been generated, skipping...")
            return
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

        print(f"Log spectro: {np.amin(log_spectro)}-{np.amax(log_spectro)}\nDynamic: {self.dynamic_min}-{self.dynamic_max}")

        color_map = plt.cm.get_cmap(self.colormap)  # .reversed()
        plt.pcolormesh(time, freq, log_spectro, cmap=color_map)
        plt.clim(vmin=self.dynamic_min, vmax=self.dynamic_max)
        # plt.colorbar()
        if adjust:
            fig.axes[0].get_xaxis().set_visible(True)
            fig.axes[0].get_yaxis().set_visible(True)
            ax.set_ylabel('Frequency (Hz)')
            ax.set_xlabel('Time (s)')
            cbar = plt.colorbar()
            cbar.set_label("Amplitude (dB)", rotation=270)
        else:            
            fig.axes[0].get_xaxis().set_visible(False)
            fig.axes[0].get_yaxis().set_visible(False)
            ax.set_frame_on(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['top'].set_visible(False)

        # Saving spectrogram plot to file
        plt.savefig(output_file, bbox_inches="tight", pad_inches=0)
        plt.close()

        os.chmod(output_file, mode=FPDEFAULT)

        print(f"Successfully generated {output_file.name}.")

        metadata_input = self.path.joinpath(
            OSMOSE_PATH.spectrogram, "adjust_metadata.csv"
        )

        metadata_output = self.path.joinpath(
            OSMOSE_PATH.spectrogram,
            f"{str(self.spectro_duration)}_{str(self.dataset_sr)}",
            f"{str(self.nfft)}_{str(self.window_size)}_{str(self.overlap)}",
            "metadata.csv",
        )
        if not metadata_output.exists():
            shutil.copyfile(metadata_input, metadata_output)
            print(f"Written {metadata_output}")
