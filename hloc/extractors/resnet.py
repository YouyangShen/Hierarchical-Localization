# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 15:30:07 2022

@author: Youyang Shen
"""
import sys
from pathlib import Path
import subprocess
import torch

from ..utils.base_model import BaseModel

loftr_path = Path(__file__).parent / '../../third_party/LoFTR'
sys.path.append(str(loftr_path))
from loftrlib import LoFTR

class _LoFTR(BaseModel):
    default_conf = {
        'model_name': 'backbone.pt',
        'checkpoint_dir': loftr_path / 'src\loftr\weights',
    }
    required_inputs = ['image']
    
    def _init(self, conf):
        model_file = conf['checkpoint_dir'] / conf['model_name']
        self.net = LoFTR(conf)
        self.net.load_state_dict(torch.load("weights/backbone.pt")['state_dict'])
        
    def _forward(self, data):
        # image = data['image']
        # image = image.flip(1)  # RGB -> BGR
        # norm = image.new_tensor([103.939, 116.779, 123.68])
        # image = (image * 255 - norm.view(1, 3, 1, 1))  # caffe normalization
        (fea_c,fea_f) = self.net(data)
        return {
            'coarse descriptors': torch.from_numpy(fea_c.T)[None],
            'fine descriptors': torch.from_numpy(fea_f.T)[None]
        }
        