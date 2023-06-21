import struct
from OSmOSE.config import SUPPORTED_AUDIO_FORMAT
from typing import Tuple

def is_audio(filename):
    return any([filename.endswith(ext) for ext in SUPPORTED_AUDIO_FORMAT])

def read_header(file: str) -> Tuple[int, float, int, int]:
    """Read the first bytes of a wav file and extract its characteristics.

    At the very least, only the first 44 bytes are read. If the `data` chunk is not right after the header chunk,
    the subsequent chunks will be read until the `data` chunk is found. If there is no `data` chunk, all the file will be read.

    Parameter
    ---------
    file: str
        The absolute path of the wav file whose header will be read.

    Returns
    -------
    samplerate : `int`
        The number of samples in one frame.
    frames : `float`
        The number of frames, corresponding to the file duration in seconds.
    channels : `int`
        The number of audio channels.
    sampwidth : `int`
        The sample width.

    Note
    ----
    When there is no `data` chunk, the `frames` value will fall back on the size written in the header. This can be incorrect,
    if the file has been corrupted or the writing process has been interrupted before completion.

    Adapted from https://gist.github.com/JonathanThorpe/9dab1729d19723ccd37730ffe477502a
    """
    with open(file, "rb") as fh:
        _, size, _ = struct.unpack("<4sI4s", fh.read(12))
        chunk_header = fh.read(8)
        subchunkid, _ = struct.unpack("<4sI", chunk_header)

        if subchunkid == b"fmt ":
            _, channels, samplerate, _, _, sampwidth = struct.unpack(
                "HHIIHH", fh.read(16)
            )

        chunkOffset = fh.tell()
        found_data = False
        while chunkOffset < size and not found_data:
            fh.seek(chunkOffset)
            subchunk2id, subchunk2size = struct.unpack("<4sI", fh.read(8))
            if subchunk2id == b"data":
                found_data = True

            chunkOffset = chunkOffset + subchunk2size + 8

        if not found_data:
            print(
                "No data chunk found while reading the header. Will fallback on the header size."
            )
            subchunk2size = size - 36

        sampwidth = (sampwidth + 7) // 8
        framesize = channels * sampwidth
        frames = subchunk2size / framesize

        if (size - 72) > subchunk2size:
            print(
                f"Warning : the size indicated in the header is not the same as the actual file size. This might mean that the file is truncated or otherwise corrupt.\
                \nSupposed size: {size} bytes \nActual size: {subchunk2size} bytes."
            )

        return samplerate, frames, channels, sampwidth
