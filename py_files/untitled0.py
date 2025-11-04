#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 29 10:17:25 2025

@author: mirandahulsey-vincent
"""

import numpy as np 

f = "/Volumes/my_own_ssd/2024_AreaX_lesions_NMA_and_sham/AreaXlesion_TweetyBERT_outputs/new_outputs/TweetyBERT_Pretrain_LLB_AreaX_FallSong_USA5509.npz"

arr = np.load(f, allow_pickle=True)
hdbscan_labelsx  = arr['hdbscan_lablels']


print(arr.files)

print(np.unique(arr["hdbscan_labels"]))
print(arr['hdbscan_labels'])