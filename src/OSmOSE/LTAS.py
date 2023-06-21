#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 19 09:17:06 2023

@author: cazaudo
"""

from OSmOSE.Spectrogram import Spectrogram
from typing import List, Tuple, Union, Literal


class LTAS(Spectrogram):
    """Main class for spectrogram-related computations. Can resample, reshape and normalize audio files before generating spectrograms."""

    def __init__(
        self,
        dataset_path: str,
        *,
        dataset_sr: int = None,
        gps_coordinates: Union[str, list, tuple] = None,
        owner_group: str = None,
        analysis_params: dict = None,
        batch_number: int = 10,
        local: bool = True,
    ) -> None:
        
        
        
        
        print(self.check_existing_matrix())