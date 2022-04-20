# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 20:03:08 2022

@author: Youyang Shen
"""

import h5py
import numpy as np
LONG = "feats-LoFTR-r1024_matches-loftr_pairs-netvlad.h5"
SHORT = 'feats-LoFTR-r1024.h5'

def update_two_h5_files(filename_short = SHORT, filename_long = LONG):
   
    print(filename_short,filename_long)
    kpts_sum = {}
    with h5py.File(filename_long, "r") as f:
        # List all groups
        keys_list = list(f.keys())
        # print(f'length of keys list is {len(keys_list)}')
        for i,item in enumerate(keys_list): # first level
            # print(f'The {i}-th item in keys list is {item}')
            
            # print(list(f[item]))
            keys_list2 = list(f[item])
            kpts0 = []
            for j,item2 in enumerate(keys_list2):       #second level     
                # print(list(f[item][item2]))
                
                keys_list3 = list(f[item][item2])
                if len(keys_list3)==3:
                    matches0 = list(f[item][item2][keys_list3[0]])
                    matches1 = list(f[item][item2][keys_list3[1]])
                    for item3 in matches0:
                        item3 = item3.tolist()
                        if item3 not in kpts0:
                            kpts0.append(item3)
            kpts_sum[item] = kpts0
              
    #second round for matches1
        
        keys_list = list(f.keys())
        
        # compliment keypoints sum with data from matches1
        for item in keys_list:
            keys_list2 = list(f[item])
            for item2 in keys_list2:
                keys_list3 = list(f[item][item2])
                matches1 = list(f[item][item2][keys_list3[1]])
                for item3 in matches1:
                    item3 = item3.tolist()
                    if item2 not in kpts_sum:
                        kpts_sum[item2] = []
                    if item3 not in kpts_sum[item2]:
                        kpts_sum[item2].append(item3)  
            
    f.close()
    print('1. keys sum finished!')
    
    #save to file
    for item, value in kpts_sum.items():
        with h5py.File(filename_short, 'a') as fd:    
            if fd.get(item):
                grp = fd.get(item)
            else:
                grp = fd.create_group(str(item)) # first level group
            
            # if fd[item]['keypoints']:
            #     # if dataset exists, delete
            #     del fd[item]['keypoints']
                
            grp.create_dataset('keypoints', data=np.array(kpts_sum[item],dtype=float))
    
    
        
    fd.close()
    print('2. keypoints updated!')

    
    #update long by long

    with h5py.File(filename_long, "r") as f:
        keys_list = list(f.keys())
        matches_sum = {}
        scores_sum = {}
        for item in keys_list:
            matches_sum[item] = {}
            scores_sum[item] = {}
            keys_list2 = list(f[item])
            for item2 in keys_list2:
                matches_sum[item][item2] = []
                scores_sum[item][item2] = []
        
                keys_list3 = list(f[item][item2])
                matches0 = list(f[item][item2][keys_list3[0]]) # list of points in A that has a match in B
                matches1 = list(f[item][item2][keys_list3[1]]) # list of points in B that has a match in A
                matching_scores0 = list(f[item][item2][keys_list3[2]]) # matching score
                keypoints_A = kpts_sum[item] # all points of A
                keypoints_B = kpts_sum[item2] # all points of B
                result = [-1] * len(keypoints_A) 
                result_score = [0] * len(keypoints_A) 
                for i in range(len(matches0)):
                    point_A = matches0[i].tolist()
                    point_B = matches1[i].tolist()
    
                    idx_A = keypoints_A.index(point_A)
                    idx_B = keypoints_B.index(point_B)
                    result[idx_A] = idx_B
                    result_score[idx_A] = matching_scores0[i]
                    
                matches_sum[item][item2] = result
                scores_sum[item][item2] = result_score

    f.close()
    print('3. matches sum and scores sum!')
   
    for item, value in matches_sum.items():
        with h5py.File(filename_long, 'a') as fd: 
            if fd[item]:
                grp = fd[item]
            else: 
                grp = fd.create_group(str(item)) # first level group 
            for item2, v in value.items():
                
                if fd[item][item2]:
                    subgrp = fd[item][item2]
                else: 
                    subgrp = grp.create_group(str(item2))
                if fd[item][item2]['matches0']:
                    del fd[item][item2]['matches0']
                if fd[item][item2]['matching_scores0']:
                    del fd[item][item2]['matching_scores0']
                    
                subgrp.create_dataset('matches0', data=matches_sum[item][item2])
                subgrp.create_dataset('matching_scores0', data=scores_sum[item][item2])  
            
    fd.close()
    print('4. matches0 and scores0 updated!')

# #%% save to h5
# file_name = 'feats-LoFTR-r1024_matches-loftr_pairs-db-covis20.h5'

# for item, value in matches_sum.items():
#     with h5py.File(str(file_name), 'a') as fd:    
#         grp = fd.create_group(str(item)) # first level group 
#         for item2, v in value.items():
#             subgrp = grp.create_group(str(item2))
#             subgrp.create_dataset('matches0', data=matches_sum[item][item2])
#             subgrp.create_dataset('matching_scores0', data=scores_sum[item][item2])  
            
#     fd.close()
    
# print('finished matching load')


#%%


def read_saved_file():
    import h5py
    LONG = "feats-LoFTR-r1024_matches-loftr_pairs-netvlad.h5"
    SHORT = 'feats-LoFTR-r1024.h5'
    
    with h5py.File(LONG, "r") as f:   
        keys_list = list(f.keys())
        for item in keys_list: # level 1
            keys_list2 = list(f[item])
            for item2 in keys_list2:
                keys_list3 = list(f[item][item2])
                matches0 = list(f[item][item2]['matches0'])
                scores0 = list(f[item][item2]['matching_scores0'])
    
    f.close()
