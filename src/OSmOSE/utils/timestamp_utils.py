from datetime import datetime, timedelta
import pandas as pd
from typing import List


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
