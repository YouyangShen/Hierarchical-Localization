# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 15:30:07 2022

@author: Youyang Shen
"""
import os
import sys
from pathlib import Path
import subprocess
import torch
from copy import deepcopy
from ..utils.base_model import BaseModel

loftr_path = Path(__file__).parent / '../../third_party/LoFTR'
sys.path.append(str(loftr_path))
from src.loftr import LoFTR 
from src.loftr import default_cfg
from configs.loftr.outdoor.buggy_pos_enc.loftr_ot import cfg
from src.loftr.utils.cvpr_ds_config import lower_config

class resnet(BaseModel):
    default_conf = deepcopy(lower_config(cfg)['loftr'])
    required_inputs = ['image']
    
    def _init(self, conf):
        # model_file = conf['checkpoint_dir'] / conf['model_name']
        # _default_cfg = deepcopy(_cfg['loftr'])
        _default_cfg = deepcopy(lower_config(cfg)['loftr'])
        self.net = LoFTR(_default_cfg)
        # self.backbone = LoFTR(*args, **kwargs)
        # print(self.net.coarse_matching.match_type)
        self.net.load_state_dict(torch.load("third_party/LoFTR/src/loftr/weights/outdoor_ot.ckpt")['state_dict'])
        self.backbone = self.net.backbone
        self.backbone.eval()
        
    def _forward(self, data):
        image = data['image']
        # image = image.flip(1)  # RGB -> BGR
        # norm = image.new_tensor([103.939, 116.779, 123.68])
        # image = (image * 255 - norm.view(3, 1, 1, 1))  # caffe normalization
        [fea_c,fea_f] = self.backbone(image)
        return {
            'feat_c': fea_c,
            'feat_f': fea_f
        }
        

# if __name__ == "__main__":
#     default_conf = deepcopy(default_cfg)
#     _cfg = lower_config(cfg)
#     default_conf2 = deepcopy(_cfg['loftr'])
