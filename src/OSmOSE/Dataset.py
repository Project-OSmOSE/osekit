import os
import stat
from pathlib import Path
from typing import Union, Tuple, List
from datetime import datetime
from warnings import warn
from statistics import fmean as mean

try:
    import grp

    skip_perms = False
except ModuleNotFoundError:
    skip_perms = True

import pandas as pd
import numpy as np
from tqdm import tqdm
from OSmOSE.utils import read_header, check_n_files, make_path, set_umask
from core.timestamps import write_timestamp
from OSmOSE.config import *


class Dataset:
    """Super class used to create dataset compatible with the rest of the package.

    A dataset is a set of audio files located in a folder whose name is the dataset name.
    The files must be in the `raw/audio/original`subfolder. and be alongside a `timestamp.csv` file, which includes
    the name of the file and the associated timestamp, in the `%Y-%m-%dT%H:%M:%S.%fZ` strftime format.

    This file can be created using the `OSmOSE.write_timestamp` function.
    """

    def __init__(
        self,
        dataset_path: str,
        *,
        gps_coordinates: Union[str, list, Tuple] = None,
        owner_group: str = None,
        original_folder: str = None,
        local: bool = True,
    ) -> None:
        """Instanciate the dataset with at least its path.

        Parameters
        ----------
        dataset_path : `str`
            The absolute path to the dataset folder. The last folder in the path will be considered as the name of the dataset.

        gps_coordinates : `str` or `list` or `Tuple`, optional, keyword-only
            The GPS coordinates of the listening location. If it is of type `str`, it must be the name of a csv file located in `data/auxiliary`,
            otherwise a list or a tuple with the first element being the latitude coordinates and second the longitude coordinates.

        owner_group : `str`, optional, keyword-only
            The name of the group using the OsmOSE package. All files created using this dataset will be accessible by the osmose group.
            Will not work on Windows.

        original_folder : `str`, optional, keyword-only
            The path to the folder containing the original audio files. It can be set right away, passed in the build() function or automatically detected.

        Example
        -------
        >>> from pathlib import Path
        >>> from OSmOSE import Dataset
        >>> dataset = Dataset(Path("home","user","my_dataset"), coordinates = [49.2, -5], owner_group = "gosmose")
        """
        self.__path = Path(dataset_path)
        self.__name = self.__path.stem
        self.owner_group = owner_group
        self.__gps_coordinates = []
        self.__local = local
        if gps_coordinates is not None:
            self.gps_coordinates = gps_coordinates

        self.__original_folder = original_folder

        self.list_abnormal_filenames = []

        if skip_perms:
            print(
                "It seems you are on a non-Unix operating system (probably Windows). The build() method will not work as intended and permission might be uncorrectly set."
            )

        pd.set_option("display.float_format", lambda x: "%.0f" % x)

    # region Properties
    @property
    def name(self):
        """str: The Dataset name. It is readonly."""
        return self.__name

    @property
    def path(self):
        """Path: The Dataset path. It is readonly."""
        return self.__path

    @property
    def original_folder(self):
        """Path: The folder containing the original audio file."""
        return (
            self.__original_folder
            if self.__original_folder
            else self._get_original_after_build()
        )

    @property
    def gps_coordinates(
        self,
    ) -> Union[Tuple[float, float], Tuple[Tuple[float, float], Tuple[float, float]]]:
        """The GPS coordinates of the listening location. First element is latitude, second is longitude.

        GPS coordinates are used to localize the dataset and required for some utilities, like the
        weather and environment utility.

        Parameter
        ---------
        coordinates: `str` or `list` or `tuple`
            If the coordinates are a string, it must be the name of a csv file located in `raw/auxiliary/`, containing two columns: 'lat' and 'long'
            Else, they can be either a list or a tuple of two float, the first being the latitude and second the longitude; or a
            list or a tuple containing two lists or tuples respectively of floats. In this case, the coordinates are not treated as a point but
            as an area.

        Returns
        -------
        The GPS coordinates as a tuple.
        """
        if not self.__gps_coordinates:
            print("This dataset has no GPS coordinates.")
        return self.__gps_coordinates

    @gps_coordinates.setter
    def gps_coordinates(
        self,
        new_coordinates: Union[
            str,
            List[float],
            List[List[float]],
            Tuple[float, float],
            Tuple[Tuple[float, float], Tuple[float, float]],
        ],
    ):
        # TODO: Allow any iterator?
        match new_coordinates:
            case str():
                csvFileArray = pd.read_csv(
                    self.path.joinpath(OSMOSE_PATH.auxiliary, new_coordinates)
                )
                self.__gps_coordinates = [np.mean(csvFileArray["lat"]), np.mean(csvFileArray["lon"])
                    #(np.min(csvFileArray["lat"]), np.max(csvFileArray["lat"])),
                    #(np.min(csvFileArray["lon"]), np.max(csvFileArray["lon"])),
                ]
            case tuple():
                self.__gps_coordinates = new_coordinates
            case list():
                if all(isinstance(coord, list) for coord in new_coordinates):
                    self.__gps_coordinates = (
                        (new_coordinates[0][0], new_coordinates[0][1]),
                        (new_coordinates[1][0], new_coordinates[1][1]),
                    )
                else:
                    self.__gps_coordinates = (new_coordinates[0], new_coordinates[1])
                # TODO : set a standard type for coordinates
                # else:
                #     raise ValueError(
                #         f"The coordinates list must contain either only floats or only sublists of two elements."
                #     )
            case _:
                raise TypeError(
                    f"GPS coordinates must be either a list of coordinates or the name of csv containing the coordinates, but {type(new_coordinates)} found."
                )

    @property
    def owner_group(self):
        """str: The Unix group able to interact with the dataset."""
        if self.__group is None:
            print(
                "The OSmOSE group name is not defined. Please specify the group name before trying to build the dataset."
            )
        return self.__group

    @owner_group.setter
    def owner_group(self, value):
        if skip_perms:
            print("Cannot set osmose group on a non-Unix operating system.")
            self.__group = None
            return
        if value:
            try:
              gid = grp.getgrnam(value).gr_gid
            except KeyError as e:
              raise KeyError(
                f"The group {value} does not exist on the system. Full error trace: {e}"
            )

        self.__group = value

    @property
    def is_built(self):
        """Checks if self.path/data/audio contains at least one folder and none called "original"."""
        metadata_path = next(
            self.path.joinpath(OSMOSE_PATH.raw_audio).rglob("metadata.csv"), None
        )
        timestamp_path = next(
            self.path.joinpath(OSMOSE_PATH.raw_audio).rglob("timestamp.csv"), None
        )
        return metadata_path and metadata_path.exists() and timestamp_path and timestamp_path.exists() and not self.path.joinpath(OSMOSE_PATH.raw_audio,"original").exists()

    # endregion

    def build(
        self,
        *,
        original_folder: str = None,
        owner_group: str = None,
        date_template: str = None,
        bare_check: bool = False,
        auto_normalization: bool = False,
        force_upload: bool = False,
    ) -> Path:
        """
        Set up the architecture of the dataset.

        This method will do the following on the dataset:
            - Gather all audio files in the  `data/audio/original/` folder.
            - Rename files containing illegal characters.
            - Identify and record files with anomalies (short duration, unreadable header...).
            - Perform data validation and automatic normalization (if turned on).
            - Rename the original folder to fit the naming scheme ([origin_audio_file_duration]_[origin_sr])
            - Write the data/audio/original/metadata.csv file.
            - Set the permission of the dataset to the osmose group.

        Parameters
        ----------
            original_folder: `str`, optional, keyword-only
                The name of the folder containing the original audio file. It is named "original" by convention.
                If none is passed, the program expects to find either only one folder in the dataset audio folder, or
                a folder named `original`
            owner_group: `str`, optional, keyword_only
                The name of the group using the osmose dataset. It will have all permissions over the dataset.
            date_template: `str`, optional, keyword_only
                the date template in strftime format. For example, `2017/02/24` has the template `%Y/%m/%d`.
                It is used to generate automatically the timestamp.csv file. Alternatively, you can call the script to create the timestamp file first.
                If no template is provided, will assume that the file already exists. In future versions, the template will be guessed automatically.
                For more information on strftime template, see https://strftime.org/.
            bare_check : `bool`, optional, keyword_only
                Only do the checks and build steps that requires low resource usage. If you build the dataset on a login node or
                if you are sure it is already good to use, set to True. Otherwise, it should be inside a job. Default is False.
            auto_normalization: `bool`, optional, keyword_only
                If true, automatically normalize audio files if the data would cause issues downstream. The default is False.
            force_upload: `bool`, optional, keyword_only
                If true, ignore the file anomalies and build the dataset anyway. The default is False.

        Returns
        -------
            dataset: `Dataset`
                The dataset object.

        Example
        -------
            >>> from pathlib import Path
            >>> from OSmOSE import Dataset
            >>> dataset = Dataset(Path("home","user","my_dataset"))
            >>> dataset.build()

            DONE ! your dataset is on OSmOSE platform !
        """
        if not self.__local:
            set_umask()
            if owner_group is None:
                owner_group = self.owner_group
    
            if not skip_perms:
                    print("\nSetting OSmOSE permission to the dataset...")
                    if owner_group:
                        gid = grp.getgrnam(owner_group).gr_gid
                        try:
                            os.chown(self.path, -1, gid)
                        except PermissionError:
                            print(f"You have not the permission to change the owner of the {self.path} folder. This might be because you are trying to rebuild an existing dataset. The group owner has not been changed.")
    
                    # Add the setgid bid to the folder's permissions, in order for subsequent created files to be created by the same user group.
                    os.chmod(self.path, DPDEFAULT)

        path_raw_audio = original_folder if original_folder is not None else self._find_or_create_original_folder()
        path_timestamp_formatted = path_raw_audio.joinpath("timestamp.csv")

        if not path_timestamp_formatted.exists():
            if not date_template:
                raise FileNotFoundError(
                    f"The timestamp.csv file has not been found in {path_raw_audio}. You can create it automatically by setting the date template as argument."
                )
            else:
                write_timestamp(audio_path=path_raw_audio, date_template=date_template)

        csvFileArray = pd.read_csv(path_timestamp_formatted, header=None)

        timestamp_csv = csvFileArray[1].values
        filename_csv = csvFileArray[0].values

        list_filename_abnormal_duration = []

        list_file_problem = []
        timestamp = []
        filename_rawaudio = []
        list_duration = []
        list_samplingRate = []
        list_interWavInterval = []
        list_size = []
        list_sampwidth = []
        list_filename = []
        lost_levels = False

        audio_file_list = [Path(path_raw_audio, indiv) for indiv in filename_csv]

        if not bare_check:
            lost_levels = check_n_files(
                audio_file_list,
                10,
                output_path=self.path.joinpath(
                    OSMOSE_PATH.raw_audio, "normalized_original"
                ),
                auto_normalization=auto_normalization,
            )

        for ind_dt in tqdm(range(len(timestamp_csv))):
            if ind_dt < len(timestamp_csv) - 1:
                diff = datetime.strptime(
                    timestamp_csv[ind_dt + 1], "%Y-%m-%dT%H:%M:%S.%fZ"
                ) - datetime.strptime(timestamp_csv[ind_dt], "%Y-%m-%dT%H:%M:%S.%fZ")
                list_interWavInterval.append(diff.total_seconds())

            audio_file = audio_file_list[ind_dt]
            list_filename.append(audio_file)

            try:
                sr, frames, sampwidth, channel_count = read_header(audio_file)

            except Exception as e:
                list_file_problem.append(audio_file)
                print(
                    f"The audio file {audio_file} could not be loaded, its importation has been canceled.\nDescription of the error: {e}"
                )
                list_filename_abnormal_duration.append(audio_file)

            list_size.append(audio_file.stat().st_size / 1e6)

            list_duration.append(frames / float(sr))
            #     list_volumeFile.append( np.round(sr * params.channel_count * (sampwidth) * frames / float(sr) /1024 /1000))
            list_samplingRate.append(float(sr))
            list_sampwidth.append(sampwidth)

            # reformat timestamp.csv
            date_obj = datetime.strptime(timestamp_csv[ind_dt], "%Y-%m-%dT%H:%M:%S.%fZ")
            dates = datetime.strftime(date_obj, "%Y-%m-%dT%H:%M:%S.%f")
            # simply chopping !
            dates_final = dates[:-3] + "Z"
            timestamp.append(dates_final)

            # we remove the sign '-' in filenames (because of our qsub_resample.sh)

            if "-" in audio_file.name:
                cur_filename = audio_file.name.replace("-", "_")
                path_raw_audio.joinpath(audio_file.name).rename(
                    path_raw_audio.joinpath(cur_filename)
                )
            else:
                cur_filename = audio_file.name
            filename_rawaudio.append(cur_filename)

        if list_filename_abnormal_duration:
            print(
                "Please see list of audio files above that canceled your dataset importation (maybe corrupted files with 0kb volume ?). You can also find it in the list list_filename_abnormal_duration, and execute following cell to directly delete them. Those filenames have been written in the file ./raw/audio/files_not_loaded.csv"
            )

            with open(path_raw_audio.joinpath("files_not_loaded.csv"), "w") as fp:
                fp.write("\n".join(str(list_filename_abnormal_duration)))
            if not force_upload:
                return list_filename_abnormal_duration

        dd = pd.DataFrame(list_interWavInterval).describe()
        print("Summary statistics on your INTER-FILE DURATION")
        print(dd[0].to_string())
        if dd[0]["std"] < 1e-10:
            dutyCycle_percent = round(
                100 * mean(list_duration) / mean(list_interWavInterval),
                1,
            )
        else:
            dutyCycle_percent = np.nan

        # get files with too small duration
        nominalVal_duration = int(np.percentile(list_duration, 10))
        print("\n Summary statistics on your file DURATION")
        dd_duration = pd.DataFrame(list_duration).describe()
        print(dd_duration[0].to_string())
        # go through the duration and check whether abnormal files
        ct_abnormal_duration = 0
        self.list_abnormal_filenames = []
        list_abnormalFilename_duration = []

        for name, duration in zip(list_filename, list_duration):
            if int(duration) < int(nominalVal_duration):
                ct_abnormal_duration += 1
                self.list_abnormal_filenames.append(name)
                list_abnormalFilename_duration.append(duration)

        if ct_abnormal_duration > 0 and not force_upload:
            print(
                "\n \n SORRY but your dataset contains files with different durations, especially",
                str(len(self.list_abnormal_filenames)),
                "files that have durations smaller than the 10th percentile of all your file durations.. \n",
            )

            print(
                "Here are their summary stats:",
                pd.DataFrame(list_abnormalFilename_duration).describe()[0].to_string(),
                "\n",
            )

            print(
                "So YOUR DATASET HAS NOT BEEN IMPORTED ON OSMOSE PLATFORM, but you have the choice now : either 1) you can force the upload using the variable force_upbload , or 2) you can first delete those files with small durations, they have been put into the variable list_abnormalFilename_name and can be removed from your dataset using the cell below"
            )
            return
        
        else:
            df = pd.DataFrame({"filename": filename_rawaudio, "timestamp": timestamp})
            df.sort_values(by=["timestamp"], inplace=True)
            df.to_csv(
                path_raw_audio.joinpath("timestamp.csv"),
                index=False,
                na_rep="NaN",
                header=None
            )
            os.chmod(path_raw_audio.joinpath("timestamp.csv"), mode=FPDEFAULT)

            # change name of the original wav folder
            new_folder_name = path_raw_audio.parent.joinpath(
                str(int(mean(list_duration))) + "_" + str(int(mean(list_samplingRate)))
            )
            
            if new_folder_name.exists():
                new_folder_name.rmdir()

            path_raw_audio = path_raw_audio.rename(new_folder_name)
            self.__original_folder = path_raw_audio

            for subpath in OSMOSE_PATH:
                if "data" in str(subpath):
                    make_path(self.path.joinpath(subpath), mode=DPDEFAULT)

            # rename filenames in the subset_files.csv if any to replace -' by '_'
            subset_path = OSMOSE_PATH.processed.joinpath("subset_files.csv")
            if subset_path.is_file():
                xx = pd.read_csv(subset_path, header=None).values
                pd.DataFrame([ff[0].replace("-", "_") for ff in xx]).to_csv(
                    subset_path,
                    index=False,
                    header=None
                )
                os.chmod(subset_path, mode=FPDEFAULT)

            # change permission on the dataset
            if ct_abnormal_duration > 0 and force_upload:
                print("\n Well you have anomalies but you choose to FORCE UPLOAD")


        # write metadata.csv
        data = {
            "origin_sr": int(mean(list_samplingRate)),
            "sample_bits": int(8 * mean(list_sampwidth)),
            "channel_count": int(channel_count),
            "audio_file_count": len(filename_csv),
            "start_date": timestamp_csv[0],
            "end_date": timestamp_csv[-1],
            "duty_cycle": dutyCycle_percent,
            "audio_file_origin_duration": round(mean(list_duration), 2),
            "audio_file_origin_volume": mean(list_size),
            "dataset_origin_volume": round(
                sum(list_size),
                1,
            ),
            "dataset_origin_duration": round(
                sum(list_duration),  # miiiiight break smth. We'll see.
                2,
            ),
            "lost_levels_in_normalization": lost_levels,
            "is_built": True,
        }
        df = pd.DataFrame.from_records([data])
        if self.gps_coordinates:
            df["lat"] = self.gps_coordinates[0]
            df["lon"] = self.gps_coordinates[1]

        df["dataset_sr"] = int(mean(list_samplingRate))
        df["audio_file_dataset_duration"] = int(round(mean(list_duration), 2))
        df.to_csv(
            path_raw_audio.joinpath("metadata.csv"),
            index=False
        )
        os.chmod(path_raw_audio.joinpath("metadata.csv"), mode=FPDEFAULT)

        print("\n DONE ! your dataset is on OSmOSE platform !")

    def delete_abnormal_files(self) -> None:
        """Delete all files with abnormal durations in the dataset, and rewrite the timestamps.csv file to reflect the changes.
        If no abnormal file is detected, then it does nothing."""

        if not self.list_abnormal_filenames:
            warn(
                "No abnormal file detected. You need to run the Dataset.build() method in order to detect abnormal files before using this method."
            )
            return

        timestamp_path = self.list_abnormal_filenames.parent.joinpath("timestamp.csv")

        csvFileArray = pd.read_csv(timestamp_path, header=None)

        for abnormal_file in self.list_abnormal_filenames:
            csvFileArray = csvFileArray.drop(
                csvFileArray[csvFileArray[0].values == abnormal_file.name].index
            )

            print(f"removing : {abnormal_file.name}")
            abnormal_file.unlink()

        csvFileArray.sort_values(by=[1], inplace=True)
        csvFileArray.to_csv(
            timestamp_path,
            index=False,
            na_rep="NaN",
            header=None
        )
        os.chmod(timestamp_path, mode=FPDEFAULT)

        print(
            "\n ALL ABNORMAL FILES REMOVED ! you can now re-run the build() method to finish importing it on OSmOSE platform"
        )

    def _find_or_create_original_folder(self) -> Path:
        """Search for the original folder or create it from existing files.

        This function does in order:
    - If there is any audio file in the top directory, consider them all original and create the data/audio/original/ directory before
        moving all audio files in it.
    - If there is only one folder in the top directory, move it to /data/audio/original.
    - If there is a folder named "original" in the raw audio path, then return it
    - If there is only one folder in the raw audio path, then return it as the original.
    If none of the above is true, then a ValueError is raised as the original folder could not be found nor created. 

        Returns
        -------
            original_folder: `Path`
                The path to the folder containing the original files.

        Raises
        ------
            ValueError
                If the original folder is not found and could not be created.
        """
        path_raw_audio = self.path.joinpath(OSMOSE_PATH.raw_audio)
        audio_files = []
        timestamp_files = []

        make_path(path_raw_audio.joinpath("original"), mode=DPDEFAULT)

        for path, _, files in os.walk(self.path):
            for f in files:
                if not Path(f).parent == "original":
                    if f.endswith((".wav",".WAV","*.mp3",".*flac")):
                        audio_files.append(Path(path,f))
                    elif "timestamp.csv" in f:
                        timestamp_files.append(Path(path,f))
        
        if len(timestamp_files) > 1:
            res = "-1"
            choice = ""
            for i, ts in enumerate(timestamp_files):
                choice += f"{i+1}: {ts}\n"
            while int(res) not in range(1,len(timestamp_files) +1):
                res = input(f"Multiple timestamp.csv detected. Choose which one should be considered the original:\n{choice}")

                timestamp_files[int(res)-1].rename(path_raw_audio.joinpath("original","timestamp.csv"))
        elif len(timestamp_files) == 1:
            timestamp_files[0].rename(path_raw_audio.joinpath("original","timestamp.csv"))

        for audio in audio_files:
            audio.rename(path_raw_audio.joinpath("original",audio.name))
            os.chmod(path_raw_audio.joinpath("original",audio.name), mode=FPDEFAULT)
            if len(os.listdir(audio.parent)) == 0:
                self.path.joinpath(audio.parent).rmdir()

        return path_raw_audio.joinpath("original")
        # if any(
        #     file.endswith(".wav") for file in os.listdir(self.path)
        # ):  # If there are audio files in the dataset folder
        #     make_path(path_raw_audio.joinpath("original"), mode=DPDEFAULT)

        #     for audiofile in os.listdir(self.path):
        #         if audiofile.endswith(".wav"):
        #             self.path.joinpath(audiofile).rename(
        #                 path_raw_audio.joinpath("original", audiofile)
        #             )
        #     return path_raw_audio.joinpath("original")
        # elif path_raw_audio.exists():
        #     if path_raw_audio.joinpath("original").is_dir():
        #         return path_raw_audio.joinpath("original")
        #     elif len(list(path_raw_audio.iterdir())) == 1:
        #         return path_raw_audio.joinpath(next(path_raw_audio.iterdir()))
        # elif (
        #     len(next(os.walk(self.path))[1]) == 1
        # ):  # If there is exactly one folder in the dataset folder
        #     make_path(path_raw_audio, mode=DPDEFAULT)
        #     orig_folder = self.path.joinpath(next(os.walk(self.path))[1][0])
        #     new_path = orig_folder.rename(path_raw_audio.joinpath(orig_folder.name))
        #     return new_path


        # else:
        #     raise ValueError(
        #         f"No folder has been found in {path_raw_audio}. Please create the raw audio file folder and try again."
        #     )

    def _get_original_after_build(self) -> Path:
        """Find the original folder path after the dataset has been built.

        Returns
        -------
            original_folder: `Path`
                The path to the folder containing the original audio file.

        Raises
        ------
            ValueError
                If no metadata.csv has been found and the original folder is not able to be found.
        """
        # First, grab any metadata.csv
        all_datasets = self.path.joinpath(OSMOSE_PATH.raw_audio).iterdir()
        while True:
            audio_dir = next(all_datasets)
            try:
                metadata_path = audio_dir.resolve().joinpath("metadata.csv")
            except StopIteration:
                # If we get to the end of the generator, it means that no metadata file has been found, so we raise a more explicit error.
                raise ValueError(
                    f"No metadata file found in {self.path.joinpath(OSMOSE_PATH.raw_audio, audio_dir)}. Impossible to find the original data file."
                )
            if metadata_path.exists():
                break

        metadata = pd.read_csv(metadata_path)
        # Catch the parameters inscribed in the original folder name
        audio_file_origin_duration = int(metadata["audio_file_origin_duration"][0])
        sr_origin = int(metadata["dataset_sr"][0])

        self.__original_folder = self.path.joinpath(
            OSMOSE_PATH.raw_audio, f"{audio_file_origin_duration}_{sr_origin}"
        )

        return self.original_folder

    def __str__(self):
        metadata = pd.read_csv(self.original_folder.joinpath("metadata.csv"))
        list_display_metadata = [
            "dataset_sr",
            "audio_file_count",
            "start_date",
            "end_date",
            "audio_file_origin_duration",
        ]  # restrain metadata to a shorter list of fileds to be displayed
        joined_str = ""
        print(f"Metadata of {self.name} :")
        for key, value in zip(metadata.keys(), metadata.values[0]):
            if key in list_display_metadata:
                joined_str += f"- {key} : {value} \n"
        return joined_str
