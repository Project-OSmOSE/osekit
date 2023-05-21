from OSmOSE.config import SUPPORTED_AUDIO_FORMAT

def is_audio(filename):
    return any([filename.endswith(ext) for ext in SUPPORTED_AUDIO_FORMAT])