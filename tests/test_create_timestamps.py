import OSmOSE.core.timestamps as tm
import re
import pytest

@pytest.mark.unit
def test_convert_template_to_re():
    raw_all = "".join(tm.__converter.keys())
    simple_template = "%Y/%m/%d"
    simple_text = "sample_file_2017/02/24.txt"
    invalid_simple_text = "sample_file_2049/25/01"
    complex_template = "y_%Y-m_%m, %I%p."
    complex_text = " y_2017-m_02, 11AM%"

    assert tm.convert_template_to_re(raw_all) == "".join(tm.__converter.values())
    simple_res = tm.convert_template_to_re(simple_template)
    assert re.search(simple_res, simple_text)[0] == "2017/02/24"
    assert re.search(simple_res, invalid_simple_text) == None
    complex_res = tm.convert_template_to_re(complex_template)
    assert re.search(complex_res, complex_text)[0] == "y_2017-m_02, 11AM%"


@pytest.mark.integ
def test_write_timestamp(tmp_path):
    true_template = "%d%m%y_%H%M%S"
    bad_template = "%Y%I%S%p"
    true_offsets = (5, 4)
    expected_result = []
    resfile = tmp_path.joinpath("timestamp.csv")

    for i in range(10):
        filename = f"test_120723_1815{str(3*i).zfill(2)}.wav"
        open(tmp_path.joinpath(filename), "w").close()
        expected_result.append(f"{filename},2023-07-12T18:15:{str(3*i).zfill(2)}.000Z\n")#,UTC TEMPORARY
    
    tm.write_timestamp(audio_path=tmp_path, date_template = true_template)

    with open(resfile, "r") as f:
        assert expected_result == f.readlines()
    resfile.unlink()

    with pytest.raises(ValueError) as excinfo:
        tm.write_timestamp(audio_path=tmp_path, date_template = bad_template)
    
    assert "The date template does not match" in str(excinfo.value)

