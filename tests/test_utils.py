from io import StringIO
import os
import pytest
import shutil
from OSmOSE.utils import *
from OSmOSE.config import OSMOSE_PATH


def test_display_folder_storage_infos(monkeypatch):
    mock_usage = namedtuple("usage", ["total", "used", "free"])
    monkeypatch.setattr(
        shutil, "disk_usage", lambda: mock_usage(2048**4, 1536**4, 1862**4)
    )

    assert True


def test_read_header(input_dir):
    sr, frames, channels, sampwidth = read_header(input_dir.joinpath("test.wav"))

    assert sr == 44100
    assert frames == 132300.0
    assert channels == 1
    assert sampwidth == 2


@pytest.mark.filterwarnings("ignore:3 NaN detected")
def test_safe_read(input_dir):
    rate = 44100  # samples per second
    duration = 3
    rng = np.random.default_rng()

    data = rng.standard_normal(duration * rate)
    sf.write(
        input_dir.joinpath("nonan.wav"), data, rate, format="WAV", subtype="DOUBLE"
    )

    assert np.array_equal(data, safe_read(input_dir.joinpath("nonan.wav"))[0])

    nandata = data.copy()
    expected = data.copy()
    nandata[0], nandata[137], nandata[2055] = np.nan, np.nan, np.nan
    expected[0], expected[137], expected[2055] = 0.0, 0.0, 0.0
    sf.write(
        input_dir.joinpath("nan.wav"), nandata, rate, format="WAV", subtype="DOUBLE"
    )

    assert np.array_equal(expected, safe_read(input_dir.joinpath("nan.wav"))[0])


def test_read_header(input_dir):
    sr = 44100
    frames = float(sr * 3)
    channels = 1
    sampwidth = 8

    assert (sr, frames, channels, sampwidth) == read_header(
        input_dir.joinpath("test.wav")
    )


def test_check_n_files_ok_files(input_dir, output_dir):
    file_list = [input_dir.joinpath("test.wav")]
    for i in range(9):
        wav_file = input_dir.joinpath(f"test{i}.wav")
        shutil.copyfile(input_dir.joinpath("test.wav"), wav_file)
        file_list.append(wav_file)

    # False returned means there was no normalization
    assert not check_n_files(file_list=file_list, n=10, output_path=output_dir)

    assert len(os.listdir(output_dir)) == 0


def test_check_n_files_bad_files(input_dir, output_dir):
    file_list = []
    rate = 44100  # samples per second
    duration = 3
    rng = np.random.default_rng()

    data = rng.standard_normal(duration * rate) + 4
    for i in range(10):
        wav_file = input_dir.joinpath(f"test{i}.wav")
        sf.write(
            input_dir.joinpath(wav_file), data, rate, format="WAV", subtype="DOUBLE"
        )
        file_list.append(wav_file)

    assert check_n_files(
        file_list=file_list, n=10, output_path=output_dir, auto_normalization=True
    )
    assert len(os.listdir(output_dir)) == 10

    shutil.rmtree(output_dir)

    assert check_n_files(
        file_list=file_list, n=5, output_path=output_dir, auto_normalization=True
    )
    assert len(os.listdir(output_dir)) == 10
    
def test_check_n_files_under_threshold_bad_files(input_dir, output_dir, monkeypatch):
    monkeypatch.setattr("sys.stdin", StringIO("y\n"))
    file_list = [input_dir.joinpath("test.wav")]
    autonorm = True if not sys.__stdin__.isatty() else False
    for i in range(8):
        wav_file = input_dir.joinpath(f"test{i}.wav")
        shutil.copyfile(input_dir.joinpath("test.wav"), wav_file)
        file_list.append(wav_file)

    rate = 44100  # samples per second
    duration = 3
    rng = np.random.default_rng()

    data = rng.standard_normal(duration * rate) - 5
    wav_file = input_dir.joinpath(f"test8.wav")
    sf.write(input_dir.joinpath(wav_file), data, rate, format="WAV", subtype="DOUBLE")
    file_list.append(wav_file)

    assert not check_n_files(
        file_list=file_list, n=10, output_path=output_dir, threshold_percent=0.1
    )
    assert len(os.listdir(output_dir)) == 0

    shutil.copyfile(wav_file, input_dir.joinpath(f"test9.wav"))
    file_list.append(input_dir.joinpath(f"test9.wav"))

    assert not check_n_files(
        file_list=file_list, n=11, output_path=output_dir, threshold_percent=0.2
    )
    assert len(os.listdir(output_dir)) == 0
    assert check_n_files(
        file_list=file_list, n=11, output_path=output_dir, threshold_percent=0.1, auto_normalization=autonorm
    )
    assert len(os.listdir(output_dir)) == 11

def test_make_path(tmp_path):
    simple_path = tmp_path / "simple_path"
    complex_path = tmp_path / "not" / "so" / "simple" / "path"
    make_path(simple_path)
    make_path(complex_path, mode=0o276)

    assert simple_path.exists()
    assert complex_path.exists()
    if sys.platform == "linux":
        assert os.stat(complex_path).st_mode == 0o276