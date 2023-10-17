import os
from PIL import Image
import glob
import numpy as np
from tqdm import tqdm
root_dir = '/remote-home/share/DATA/BCSS/Patches_nooverlap/label_grouping'
cls_set = set()
mask_names=glob.glob('/remote-home/share/DATA/BCSS/Patches_nooverlap/label/*.png')
for mask_name in tqdm(mask_names):
    mask = Image.open(mask_name)
    mask_name_base = mask_name.split("/")[-1]
    mask = np.array(mask)
    grouping_mask = np.zeros_like(np.array(mask)) 
    grouping_mask[mask==0] = 255
    grouping_mask[mask==1] = 1
    grouping_mask[mask==2] = 2
    grouping_mask[mask==3] = 3
    grouping_mask[mask==4] = 4
    grouping_mask[mask==10] = 3
    grouping_mask[mask==11] = 3 
    grouping_mask[mask==14] = 3 
    grouping_mask[mask==19] = 1
    grouping_mask[mask==20] = 1
    grouping_mask.astype(np.uint8)
    grouping_mask = Image.fromarray(grouping_mask)
    grouping_mask.save(os.path.join(root_dir, mask_name_base))
