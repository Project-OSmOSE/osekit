from pathlib import Path
from collections import namedtuple
import stat

SUPPORTED_AUDIO_FORMAT = ["wav"]

# Dict are easier to modify, namedtuple easier to use
__global_path_dict = {
    "raw_audio": Path("data", "audio"),
    "auxiliary": Path("data", "auxiliary"),
    "instrument": Path("data", "auxiliary", "instrument"),
    "environment": Path("data", "auxiliary", "environment"),
    "processed": Path("processed"),
    "spectrogram": Path("processed", "spectrogram"),
    "statistics": Path("processed", "dataset_statistics"),
}

OSMOSE_PATH = namedtuple("path_list", __global_path_dict.keys())(**__global_path_dict)

FPDEFAULT = 0o664 # Default file permissions
DPDEFAULT = (stat.S_ISGID | 0o775) # Default directory permissions