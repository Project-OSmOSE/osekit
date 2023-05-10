import os
from pathlib import Path

import pytest
from OSmOSE import Dataset


def test_init(input_dataset, capsys):
    dataset = Dataset(dataset_path=input_dataset["main_dir"])

    assert dataset.name == "sample_dataset"

    dataset.gps_coordinates
    captured = capsys.readouterr()
    assert "This dataset has no GPS coordinates." in captured.out


def test_find_or_create_original_folder(input_dataset):
    dataset = Dataset(input_dataset["main_dir"])
    folder = dataset._find_or_create_original_folder()

    assert folder == input_dataset["orig_audio_dir"]

    input_dataset["orig_audio_dir"].rename(
        input_dataset["orig_audio_dir"].with_name("unconventional_name")
    )

    folder2 = dataset._find_or_create_original_folder()

    assert folder2 == input_dataset["orig_audio_dir"].with_name("original")


def test_error_build(input_dir: Path):
    input_dir.joinpath("data", "audio").mkdir(parents=True)
    dataset = Dataset(input_dir)

    with pytest.raises(FileNotFoundError) as e:
        dataset.build()
    assert (
        str(e.value)
        == f"""The timestamp.csv file has not been found in {input_dir.joinpath("data","audio","original")}. You can create it automatically by setting the date template as argument."""
    )


def test_build(input_dataset):
    dataset = Dataset(input_dataset["main_dir"])

    dataset.build()

    new_expected_path = dataset.path.joinpath("data", "audio", "3_44100")

    assert not input_dataset["orig_audio_dir"].exists()
    assert new_expected_path.exists()

    assert sorted(os.listdir(new_expected_path)) == sorted(
        ["metadata.csv", "timestamp.csv"] + [f"test_{i}.wav" for i in range(10)]
    )


# TODO : test with broken files
