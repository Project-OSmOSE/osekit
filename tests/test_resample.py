import os
import platform
import soundfile as sf
import shutil
from OSmOSE.cluster.resample import resample
import pytest


@pytest.mark.skipif(platform.system() == "Windows", reason="Sox is linux only")
def test_resample(input_dir, output_dir):
    for i in range(3):
        wav_file = os.path.join(input_dir, f"test{i}.wav")
        shutil.copyfile(os.path.join(input_dir, "test.wav"), wav_file)

    for sr in [100, 500, 8000]:
        resample(input_dir=input_dir, output_dir=output_dir, target_fs=sr)

        # check that all resampled files exist and have the correct properties
        for i in range(3):
            output_file = os.path.join(output_dir, f"test{i}.wav")
            assert os.path.isfile(output_file)
            assert sf.info(output_file).sample_rate == sr
            assert sf.info(output_file).channels == 1
            assert sf.info(output_file).frames == 900
            assert sf.info(output_file).duration == 3.0

        assert len(os.listdir(output_dir)) == 4
        # check that the original files were not modified
        for i in range(3):
            input_file = os.path.join(input_dir, f"test{i}.wav")
            assert sf.info(input_file).sample_rate == 300