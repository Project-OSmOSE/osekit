import os
from filelock import FileLock
from typing import Literal, Union
import shutil
from OSmOSE.config import *
from termcolor import colored
from OSmOSE.features import Welch
from OSmOSE.utils import set_umask, make_path
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

class Aplose(Welch):
    def __init__(self, dataset_path: str, *, dataset_sr: int = None, gps_coordinates: str | list | tuple = None, owner_group: str = None, analysis_params: dict = None, batch_number: int = 10, local: bool = False) -> None:
        super().__init__(dataset_path, dataset_sr=dataset_sr, gps_coordinates=gps_coordinates, owner_group=owner_group, analysis_params=analysis_params, batch_number=batch_number, local=local)

        self.colormap: str = (
            self.__analysis_sheet["colormap"][0] if "colormap" in self.__analysis_sheet else "viridis"
        )
        self.zoom_level: int = (
            self.__analysis_sheet["zoom_level"][0] if "zoom_level" in self.__analysis_sheet else 0
        )
        self.dynamic_min: int = (
            self.__analysis_sheet["dynamic_min"][0]
            if "dynamic_min" in self.__analysis_sheet
            else None
        )
        self.dynamic_max: int = (
            self.__analysis_sheet["dynamic_max"][0]
            if "dynamic_max" in self.__analysis_sheet
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
        self.__build_path(dry=True)

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
        if not save_image and not save_matrix:
            raise ValueError("Neither image or matrix are set to be generated. Please set at least one of save_matrix or save_image to True to proceed with the spectrogram generation, or use the welch() method to get the raw data.")

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
        except: 
            pass

        self.save_matrix = save_matrix
        self.save_image = save_image
        
        self.adjust = adjust
        output_file = self.path_output_spectrogram.joinpath(audio_file)

        def check_existing_matrix():
            return len(list(self.path_output_spectrogram_matrix.glob(f"{Path(audio_file).stem}*"))) == 2**self.zoom_level if save_matrix else True

        if len(list(self.path_output_spectrogram.glob(f"{Path(audio_file).stem}*"))) == sum(2**i for i in range(self.zoom_level+1)) and check_existing_matrix():
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


        lock = FileLock(str(output_file) + ".lock")
        lock.acquire(blocking=False)

        self.process_file(audio_file=audio_file,
                          adjust=adjust,
                          last_file_behavior=last_file_behavior,
                          merge_files=merge_files,
                          write_audio_file=write_audio_file)
        
        lock.release()
        os.remove(str(output_file) + ".lock")


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

        duration = len(data) / int(sample_rate)

        nber_tiles_lowest_zoom_level = 2 ** (self.zoom_level)
        tile_duration = duration / nber_tiles_lowest_zoom_level

        Sxx_2 = np.empty((int(self.nfft / 2) + 1, 1))
        for tile in range(0, nber_tiles_lowest_zoom_level):
            start = tile * tile_duration
            end = start + tile_duration

            sample_data = data[int(start * sample_rate) : int((end + 1) * sample_rate)]

            Sxx, Freq = self.gen_spectro(
                data=sample_data,
                sample_rate=sample_rate,
                output_file=output_file.parent.joinpath(
                    f"{output_file.stem}_{nber_tiles_lowest_zoom_level}_{str(tile)}.png"
                ),
            )            
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

            # TODO: add an option to force regeneration (in case of corrupted file)
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

            Sxx_2 = np.hstack((Sxx_2, Sxx))

        Sxx_lowest_level = Sxx_2[:, 1:]

        segment_times = np.linspace(
            0, len(data) / sample_rate, Sxx_lowest_level.shape[1]
        )[np.newaxis, :]

        self.time_resolution[0] = segment_times[1] - segment_times[0]

        if self.save_image:
            # loop over the zoom levels from the second lowest to the highest one
            for zoom_level in range(self.zoom_level + 1)[::-1]:
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
                        adjust=self.adjust
                    )

    def initialize(
        self,
        *,
        dataset_sr: int = None,
        batch_ind_min: int = 0,
        batch_ind_max: int = -1,
        force_init: bool = False,
        date_template: str = None
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
        self.__build_path()
        super().initialize(batch_ind_min=batch_ind_min, batch_ind_max=batch_ind_max, force_init=force_init, date_template=date_template)

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

        # Load variables from raw metadata
        metadata = pd.read_csv(self.path_input_audio_file.joinpath("metadata.csv"))
        audio_file_origin_duration = metadata["audio_file_origin_duration"][0]
        sr_origin = metadata["dataset_sr"][0]
        audio_file_count = metadata["audio_file_count"][0]

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

        self.__analysis_sheet = pd.DataFrame.from_records([data])

        adjust_path = self.path.joinpath(OSMOSE_PATH.spectrogram, "adjust_metadata.csv")
        if adjust_path.exists():
            adjust_path.unlink() # Always overwrite this file

        self.__analysis_sheet.to_csv(
            adjust_path
        )
        os.chmod(adjust_path, mode=FPDEFAULT)
        
        if not adjust_path.exists():
            raise FileNotFoundError("Failed to write adjust_metadata.csv")

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
            plt.colorbar()
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
        # try:
        #     if not self.adjust and metadata_input.exists() and not metadata_output.exists():
        #         metadata_input.rename(metadata_output)
        # except:
        #     pass