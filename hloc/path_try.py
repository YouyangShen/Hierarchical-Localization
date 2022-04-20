# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 16:07:30 2022

@author: Youyang Shen
"""

import os
from pathlib import Path

feature = "outputs/aachen/feats-LoFTR-r1024.h5"

feature_path = Path(feature)

feature_path.exists()