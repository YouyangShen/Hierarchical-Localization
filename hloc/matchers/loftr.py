# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 11:39:42 2022

@author: Youyang Shen
"""

import os
import sys
from pathlib import Path
import subprocess
import torch
from copy import deepcopy
from ..utils.base_model import BaseModel
from einops.einops import rearrange
loftr_path = Path(__file__).parent / '../../third_party/LoFTR'
sys.path.append(str(loftr_path))
from src.loftr import LoFTR 
from src.loftr import default_cfg

class resnet(BaseModel):
    default_conf = deepcopy(default_cfg)
    required_inputs = ['image0','feat_c0','feat_f0',
                       'image1','feat_c1','feat_f1']
    
    def _init(self, conf):
        # model_file = conf['checkpoint_dir'] / conf['model_name']
        _default_cfg = deepcopy(default_cfg)
        self.net = LoFTR(_default_cfg)
        # self.backbone = LoFTR(*args, **kwargs)

        self.net.load_state_dict(torch.load("third_party/LoFTR/src/loftr/weights/outdoor_ds.ckpt")['state_dict'])
        self.pos_encoding = self.net.pos_encoding
        self.loftr_coarse =  self.net.loftr_coarse
        self.coarse_matching = self.net.coarse_matching
        self.fine_preprocess =  self.net.fine_preprocess
        self.loftr_fine =  self.net.loftr_fine
        self.fine_matching =  self.net.fine_matching
        
    def _forward(self, data):
        # image = data['image']
        # image = image.flip(1)  # RGB -> BGR
        # norm = image.new_tensor([103.939, 116.779, 123.68])
        # image = (image * 255 - norm.view(1, 3, 1, 1))  # caffe normalization
         # 2. coarse-level loftr module
        # add featmap with positional encoding, then flatten it to sequence [N, HW, C]
        data.update({
            'bs': data['image0'].size(0),
            'hw0_i': data['image0'].shape[2:], 'hw1_i': data['image1'].shape[2:]
        })
        data.update({
            'hw0_c': data['feat_c0'].shape[2:], 'hw1_c': data['feat_c1'].shape[2:],
            'hw0_f': data['feat_f0'].shape[2:], 'hw1_f': data['feat_f1'].shape[2:]
        })
        
        feat_c0 = rearrange(self.pos_encoding(data['feat_c0']), 'n c h w -> n (h w) c')
        feat_c1 = rearrange(self.pos_encoding(data['feat_c1']), 'n c h w -> n (h w) c')

        mask_c0 = mask_c1 = None  # mask is useful in training
        if 'mask0' in data:
            mask_c0, mask_c1 = data['mask0'].flatten(-2), data['mask1'].flatten(-2)
        feat_c0, feat_c1 = self.loftr_coarse(feat_c0, feat_c1, mask_c0, mask_c1)

        # 3. match coarse-level
        self.coarse_matching(feat_c0, feat_c1, data, mask_c0=mask_c0, mask_c1=mask_c1)

        # 4. fine-level refinement
        feat_f0_unfold, feat_f1_unfold = self.fine_preprocess(data['feat_f0'], data['feat_f1'], feat_c0, feat_c1, data)
        if feat_f0_unfold.size(0) != 0:  # at least one coarse level predicted
            feat_f0_unfold, feat_f1_unfold = self.loftr_fine(feat_f0_unfold, feat_f1_unfold)

        # 5. match fine-level
        self.fine_matching(feat_f0_unfold, feat_f1_unfold, data)
        
        return {'matches0': data["mkpts0_f"],
                'matches1': data["mkpts1_f"],
                'matching_scores0': [data['mconf']]}
        

        