import os
from filelock import FileLock
from typing import Literal, Union
import shutil
from OSmOSE.config import *
from OSmOSE.features import Welch
from OSmOSE.utils import set_umask, make_path
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

class Aplose(Welch):
    def __init__(self, dataset_path: str, *, dataset_sr: int = None, gps_coordinates: str | list | tuple = None, owner_group: str = None, analysis_params: dict = None, batch_number: int = 10, local: bool = False) -> None:
        super().__init__(dataset_path, dataset_sr=dataset_sr, gps_coordinates=gps_coordinates, owner_group=owner_group, analysis_params=analysis_params, batch_number=batch_number, local=local)

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