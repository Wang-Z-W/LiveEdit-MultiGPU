import json
import os
import shutil
from tqdm import tqdm
from PIL import Image
import numpy as np


def _get_rephrase_image(image_path, reg_to_attr, reg_coord, rephrase_path):
    def find_digit_region(reg_to_attr):
        for i, attr in enumerate(reg_to_attr):
            if attr and attr not in ["empty", "rectangle"]:
                return i
        return None
    def extract_digit_region(image_path, region_coords, digit_region_idx):
        if digit_region_idx is None:
            return None
        img = Image.open(image_path)
        img_array = np.array(img)
        x1, y1, x2, y2 = region_coords[digit_region_idx]
        digit_region = img_array[y1:y2, x1:x2]
        digit_img = Image.fromarray(digit_region)
        return digit_img

    digit_img = extract_digit_region(
        image_path,
        reg_coord,
        find_digit_region(reg_to_attr)
    )
    
    if digit_img is None:
        return None
    digit_img.save(rephrase_path)

def rephrase_image(ann_path):
    anns = json.load(open(ann_path, 'r'))
    for ann in tqdm(anns, total=len(anns), desc="Rephrase images for {}".format(ann_path)):
        image_path = ann['image_filepath']
        reg_to_attr = ann['reg_to_attr']
        reg_coord = ann['region_coord']
        rephrase_path = image_path.replace('.png', '_rephrase.png')
        _get_rephrase_image(image_path, reg_to_attr, reg_coord, rephrase_path)
        
        
rephrase_image("spumnist_base/annotations_train.json")
rephrase_image("spumnist_base/annotations_test.json")
