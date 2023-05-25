from OSmOSE.config import SUPPORTED_AUDIO_FORMAT, UNSUPPORTED_AUDIO_FORMAT

class BadExtensionError(Exception):
    """General class to catch unsupported or unknown file extensions"""

class UnknownAudioFormatError(BadExtensionError):
    def __init__(self, ext:str = ""):
        self.ext = ext
    def __str__(self):
        return f"""The audio file format {self.ext + " "}is unknown. Supported formats are {",".join(SUPPORTED_AUDIO_FORMAT)}."""

class UnsupportedAudioFormatError(BadExtensionError):
    def __init__(self, ext:str = ""):
        self.ext = ext
    def __str__(self):
        return f"""The audio file format {self.ext + " "}is not supported by OSmOSE. Supported formats are {",".join(SUPPORTED_AUDIO_FORMAT)}."""