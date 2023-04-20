import re
import os
import datetime
import argparse
import pandas as pd
from pathlib import Path

from OSmOSE.config import *

__converter = {
    "%Y": r"[12][0-9]{3}",
    "%y": r"[0-9]{2}",
    "%m": r"(0[1-9]|1[0-2])",
    "%d": r"([0-2][0-9]|3[0-1])",
    "%H": r"([0-1][0-9]|2[0-4])",
    "%I": r"(0[1-9]|1[0-2])",
    "%p": r"(AM|PM)",
    "%M": r"[0-5][0-9]",
    "%S": r"[0-5][0-9]",
    "%f": r"[0-9]{6}",
}


def convert_template_to_re(date_template: str) -> str:
    """
    Converts a template in strftime format to a matching regular expression.

    Parameter:
        date_template: the template in strftime format.

    Returns:
        The regular expression matching the template.
    """

    res = ""
    i = 0
    while i < len(date_template):
        if date_template[i : i + 2] in __converter:
            res += __converter[date_template[i : i + 2]]
            i += 1
        else:
            res += date_template[i]
        i += 1

    return res


def write_timestamp(
    *,
    audio_path: str,
    date_template: str,
    timezone: str = "UTC",
    offsets: tuple = None,
):
    """
    Read the dates in the filenames of audio files in the `audio_path` folder,
    according to the date template in strftime format or the offsets from the beginning and end of the date.

    If both `date_template` and `offsets` are provided, the latter has priority and `date_template` is ignored.

    The result is written in a file named `timestamp.csv` with no header and two columns in this format : [filename],[timestamp].
    The output date is in the template `'%Y-%m-%dT%H:%M:%S.%fZ'.

    Parameters
    ----------
        audio_path: `str`
            the path of the folder containing audio files
        date_template: `str`
            the date template in strftime format. For example, `2017/02/24` has the template `%Y/%m/%d`
            For more information on strftime template, see https://strftime.org/
        timezone: `str`, optional
            The timezone this timestamp was originally recorded in (the default is UTC).
        offsets: `tuple(int,int)`, optional
            a tuple containing the beginning and end offset of the date.
            The first element is the first character of the date, and the second is the last.
    """
    if offsets:
        if "-" in offsets:
            offset = [int(off) for off in offsets.split("-")]
        else:
            offset = [int(offsets), 0]
    # TODO: extension-agnostic
    list_audio_file = sorted([file for file in Path(audio_path).glob("*.wav")])

    if len(list_audio_file) == 0:
        print(
            f"No audio file found in the {audio_path} directory. An empty timestamp.csv will be created."
        )

    timestamp = []
    filename_raw_audio = []

    converted = convert_template_to_re(date_template)
    for filename in list_audio_file:
        if offsets:
            date_extracted = filename.stem[offset[0] : offset[1] + 1]
        else:
            try:
                date_extracted = re.search(converted, str(filename))[0]
            except TypeError:
                raise ValueError(
                    f"The date template does not match any set of character in the file name {filename}\nMake sure you are not forgetting separator characters, or use the offsets parameter."
                )

        date_obj = datetime.datetime.strptime(date_extracted, date_template)
        dates = datetime.datetime.strftime(date_obj, "%Y-%m-%dT%H:%M:%S.%f")

        dates_final = dates[:-3] + "Z"

        print("filename->", filename)
        print("extracted timestamp->", dates_final, "\n")

        timestamp.append(dates_final)

        filename_raw_audio.append(filename.name)

    df = pd.DataFrame(
        {"filename": filename_raw_audio, "timestamp": timestamp, "timezone": timezone}
    )
    df.sort_values(by=["timestamp"], inplace=True)
    df.to_csv(
        Path(audio_path, "timestamp.csv"),
        index=False,
        na_rep="NaN",
        header=None
    )
    os.chmod(Path(audio_path, "timestamp.csv"), mode=FPDEFAULT)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--dataset-name", "-n", help="Name of the dataset.")
    argparser.add_argument(
        "--offset",
        "-s",
        help="Offset of the first date character in the dataset names. If the date is not immediately followed by the extension, please provide the offset between the end of the date and the extension of the file, separated by a hyphen (-).",
    )
    argparser.add_argument(
        "--date-template",
        "-d",
        help="The date template in strftime format. If not sure, input the whole file name.",
    )
    argparser.add_argument(
        "--timezone",
        "-t",
        default="UTC",
        help="The timezone the date was recorded in. Default is UTC.",
    )
    args = argparser.parse_args()

    write_timestamp(
        audio_path=args.dataset_name,
        date_template=args.date_template,
        timezone=args.timezone,
        offsets=args.offset,
    )
