import os
import json
import sys
import pandas as pd
import argparse


water_birds_list = [
    'Albatross', # Seabirds
    'Auklet',
    'Cormorant',
    'Frigatebird',
    'Fulmar',
    'Gull',
    'Jaeger',
    'Kittiwake',
    'Pelican',
    'Puffin',
    'Tern',
    'Gadwall', # Waterfowl
    'Grebe',
    'Mallard',
    'Merganser',
    'Guillemot',
    'Pacific_Loon'
]


def get_groups(cub_anns):
    groups = {}
    group_factor_dict = {}
    total = 0
    for ann in cub_anns:
        total += 1
        img_path = ann['image_path']
        bird_type = 'water' if img_path.split('/')[0].split('_')[-1] in water_birds_list else 'land'
        back = 'water' if ann['water_background'].lower() == 'yes' else 'land'
        group_key = f"{bird_type}_on_{back}"
        if group_key not in groups:
            groups[group_key] = []
        groups[group_key].append(ann)
    print(f"Group numbers")
    print(f"    total: {total}")
    for key in groups.keys():
        print(f"    {key}: {len(groups[key])}")
        group_factor_dict[key] = round(len(groups[key]) / total, 4)
    return groups, group_factor_dict

def get_controled_dataset(size = None, group_factor = None):
    group_factor_dict = {
        "water_on_water": group_factor[0],
        "water_on_land": group_factor[1],
        "land_on_land": group_factor[2],
        "land_on_water": group_factor[3],
    }
    
    anns = []
    for key, factor in group_factor_dict.items():
        bird = key.split("_on_")[0]
        back = key.split("_on_")[1]
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=None)
    parser.add_argument("--group_factor", type=float, nargs=4, default=None)
    args = parser.parse_args()
    
    cub_back_anns = json.load(open("../CUB_200_2011/CUB_200_2011/water_background.json"))
    cub_groups, cub_group_factor = get_groups(cub_back_anns)
    
    import pdb; pdb.set_trace()