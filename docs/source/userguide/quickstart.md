# Quickstart

## Installation


## Basic Use

### Build a Dataset

To work properly, OSmOSE requires a specific worktree which is flexible within limits. Luckily, it is able to create a workspace in a non-disruptive way. Transforming a folder containing audio files to an OSmOSE dataset is often the first step of any analysis.

Assuming the folder my_dataset exists and contains only audio files with their date (year, month, day, hour, minute and seconds) included in their names. If that is not the case, you will need to first Generate timestamps.

```python
from OSmOSE import Dataset
from pathlib import Path

path_to_dataset = Path("/home","all_datasets", "my_dataset")
owner_group = "gosmose" # linux-only

dataset = Dataset(dataset_path=path_to_dataset, owner_group=owner_group)

date_template = "%d%m%y_%H%M%S" # the date as it formatted in the audio file names

dataset.build(date_template=date_template)
```