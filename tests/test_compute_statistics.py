import csv
import numpy as np
import soundfile as sf
from pathlib import Path
from OSmOSE.features import compute_stats


def test_output_file_written(input_dataset: Path, output_dir: Path):
    output_file = output_dir.joinpath("output.csv")
    print(output_file)
    compute_stats(input_dir=input_dataset["orig_audio_dir"], output_file=output_file, hp_filter_min_freq=100)

    assert output_file.exists()
    assert output_file.stat().st_size > 0


def test_output_file_columns(input_dataset, output_dir: Path):
    output_file = output_dir.joinpath("output.csv")
    # Call the function to write normalization parameters
    compute_stats(
        input_dir=input_dataset["orig_audio_dir"], output_file=output_file, hp_filter_min_freq=100
    )
    # Check that the output file has the correct columns
    with open(str(output_file), "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        assert header == ["filename", "timestamp", "mean", "std"]


def test_output_file_content(input_dataset, output_dir):
    output_file = output_dir / "output.csv"
    # Create two dummy WAV files with different means and stds
    wav_file1 = input_dataset["orig_audio_dir"] / "test-1.wav"
    data1 = np.random.randn(1000) + 1
    sf.write(wav_file1, data1, 44100)
    wav_file2 = input_dataset["orig_audio_dir"] / "test-2.wav"
    data2 = np.random.randn(1000) * 2
    sf.write(wav_file2, data2, 44100)
    # Call the function to write normalization parameters
    compute_stats(
        input_dir=str(input_dataset["orig_audio_dir"]), output_file=str(output_file), hp_filter_min_freq=0
    )
    # Check that the output file has the correct content
    with open(str(output_file), "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        assert header == ["filename", "timestamp", "mean", "std"]
        row1 = next(reader)
        assert row1[0] == "test-0.wav"
        # assert float(row1[1]) == pytest.approx(0, abs=1e-2)
        # assert float(row1[2]) == pytest.approx(1, abs=1e-2)
        row2 = next(reader)
        assert row2[0] == "test-1.wav"
        # assert float(row2[1]) == pytest.approx(1, abs=1e-2)
        # assert float(row2[2]) == pytest.approx(1, abs=1e-2)
        row3 = next(reader)
        assert row3[0] == "test-2.wav"
        # assert float(row3[1]) == pytest.approx(0, abs=1e-2)
        # assert float(row3[2]) == pytest.approx(2, abs=1e-2)
        # TODO: understand butterworth filter and see if it is possible to approx std
