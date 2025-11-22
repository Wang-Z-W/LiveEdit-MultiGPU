import os
import numpy as np
import random
import pandas as pd
import json
from PIL import Image
from tqdm import tqdm
from dataset_utils import crop_and_resize, combine_and_mask
import pyrootutils
import shutil

root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True)

################ Paths and other configs - Set these #################################
cub_dir = root / 'data' / 'CUB_200_2011'
places_dir = '/data/wzw/group_DRO/Places365'
output_dir = root / 'data' / 'WaterBird'
dataset_name = 'waterbird_land2water2_worstgroup_LL'

target_places = [
    ['bamboo_forest', 'forest/broadleaf'],  # Land backgrounds
    ['ocean', 'lake/natural']]              # Water backgrounds

val_frac = 0.2             # What fraction of the training data to use as validation
confounder_strength = 0.95 # Legacy helper, used below to set defaults
random_seed = 2025         # Seed for reproducibility

# Desired relative weights for each (y, c) combination in every split.
# Keys are split names: 'train', 'val', 'test'. Values are dicts mapping
# (label y, place c) -> non-negative weight. Within each split, the weights for
# a given y are normalised to match the number of samples of that label.
split_group_ratios = {
    'train': {
        (0, 0): 1-confounder_strength,
        (0, 1): confounder_strength,
        (1, 0): 0.5,
        (1, 1): 0.5,
    },
    'val': {
        (0, 0): 0.5,
        (0, 1): 0.5,
        (1, 0): 0.5,
        (1, 1): 0.5,
    },
    'test': {
        (0, 0): 0.5,
        (0, 1): 0.5,
        (1, 0): 0.5,
        (1, 1): 0.5,
    },
}
######################################################################################

np.random.seed(random_seed)
random.seed(random_seed)

output_subfolder = os.path.join(output_dir, dataset_name)
if os.path.exists(output_subfolder):
    print(f"Dataset directory '{output_subfolder}' already exists. Skipping generation.")
    raise SystemExit

images_path = os.path.join(cub_dir, 'CUB_200_2011', 'images.txt')

df = pd.read_csv(
    images_path,
    sep=" ",
    header=None,
    names=['img_id', 'img_filename'],
    index_col='img_id')

### Load water background annotations
water_background_path = os.path.join(cub_dir, 'CUB_200_2011', 'water_background.json')
if os.path.exists(water_background_path):
    with open(water_background_path, 'r') as f:
        water_background_annotations = json.load(f)
    print(f"Loaded water background annotations for {len(water_background_annotations)} images")
else:
    print("Warning: water_background.json not found. All images will be processed with background replacement.")
    water_background_annotations = {}
water_background_annotations = [ann['image_path'] for ann in water_background_annotations if ann['water_background'].lower() == 'yes']

### Set up labels of waterbirds vs. landbirds
# We consider water birds = seabirds and waterfowl.
species = np.unique([img_filename.split('/')[0].split('.')[1].lower() for img_filename in df['img_filename']])
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

water_birds = {}
for species_name in species:
    water_birds[species_name] = 0
    for water_bird in water_birds_list:
        if water_bird.lower() in species_name:
            water_birds[species_name] = 1
species_list = [img_filename.split('/')[0].split('.')[1].lower() for img_filename in df['img_filename']]
df['y'] = [water_birds[species] for species in species_list]

### Assign train/tesst/valid splits
# In the original CUB dataset split, split = 0 is test and split = 1 is train
# We want to change it to
# split = 0 is train,
# split = 1 is val,
# split = 2 is test

train_test_df =  pd.read_csv(
    os.path.join(cub_dir, 'CUB_200_2011', 'train_test_split.txt'),
    sep=" ",
    header=None,
    names=['img_id', 'split'],
    index_col='img_id')

df = df.join(train_test_df, on='img_id')
test_ids = df.loc[df['split'] == 0].index
train_ids = np.array(df.loc[df['split'] == 1].index)
val_ids = np.random.choice(
    train_ids,
    size=int(np.round(val_frac * len(train_ids))),
    replace=False)

df.loc[train_ids, 'split'] = 0
df.loc[val_ids, 'split'] = 1
df.loc[test_ids, 'split'] = 2


def _validate_split_group_ratios(config):
    required_keys = {(y, c) for y in (0, 1) for c in (0, 1)}
    for split_name, ratios in config.items():
        missing = required_keys - ratios.keys()
        if missing:
            raise ValueError(f"Split '{split_name}' is missing ratios for: {sorted(missing)}")
        for key, value in ratios.items():
            if value < 0:
                raise ValueError(f"Ratio for split '{split_name}', group {key} must be non-negative")
        for y in (0, 1):
            denom = ratios[(y, 0)] + ratios[(y, 1)]
            if denom <= 0:
                raise ValueError(f"Ratios for split '{split_name}', y={y} must have positive sum")


def _compute_group_counts(total, weights):
    weights = np.asarray(weights, dtype=float)
    weights_sum = weights.sum()
    if weights_sum <= 0:
        raise ValueError("Weights must sum to a positive value")
    expected = weights / weights_sum * total
    floor_counts = np.floor(expected).astype(int)
    remainder = int(total - floor_counts.sum())
    if remainder > 0:
        remainders = expected - floor_counts
        order = np.argsort(-remainders)
        for idx in order[:remainder]:
            floor_counts[idx] += 1
    return floor_counts


### Assign confounders (place categories)
_validate_split_group_ratios(split_group_ratios)

split_name_lookup = {0: 'train', 1: 'val', 2: 'test'}
df['place'] = -1
split_statistics = {}

for split_value, split_name in split_name_lookup.items():
    if split_name not in split_group_ratios:
        raise ValueError(f"Missing configuration for split '{split_name}'")
    ratios = split_group_ratios[split_name]

    split_indices = df.index[df['split'] == split_value]
    split_df = df.loc[split_indices]

    for y in (0, 1):
        y_indices = split_df.index[split_df['y'] == y]
        if len(y_indices) == 0:
            continue
        desired_counts = _compute_group_counts(
            len(y_indices),
            [ratios[(y, 0)], ratios[(y, 1)]]
        )
        n_place0, n_place1 = desired_counts.tolist()

        df.loc[y_indices, 'place'] = 0
        if n_place1 > 0:
            chosen = np.random.choice(y_indices, size=n_place1, replace=False)
            df.loc[chosen, 'place'] = 1

if (df['place'] == -1).any():
    raise RuntimeError("Some samples were not assigned a place label")

for split, split_label in [(0, 'train'), (1, 'val'), (2, 'test')]:
    print(f"{split_label}:")
    split_df = df.loc[df['split'] == split, :]
    total_split = len(split_df)
    group_counts = {
        (y, c): int(np.sum((split_df['y'] == y) & (split_df['place'] == c)))
        for y in (0, 1) for c in (0, 1)
    }
    split_statistics[split_label] = {
        'total': total_split,
        'waterbird_fraction': float(np.mean(split_df['y'])) if total_split > 0 else 0.0,
        'group_counts': {
            f"y{y}_c{c}": {
                'count': group_counts[(y, c)],
                'fraction_of_split': float(group_counts[(y, c)] / total_split) if total_split > 0 else 0.0
            }
            for y in (0, 1) for c in (0, 1)
        }
    }
    print(f"    waterbirds are {np.mean(split_df['y']):.3f} of the examples")
    print(f"    y = 0, c = 0: {np.mean(split_df.loc[split_df['y'] == 0, 'place'] == 0):.3f}, n = {np.sum((split_df['y'] == 0) & (split_df['place'] == 0))}")
    print(f"    y = 0, c = 1: {np.mean(split_df.loc[split_df['y'] == 0, 'place'] == 1):.3f}, n = {np.sum((split_df['y'] == 0) & (split_df['place'] == 1))}")
    print(f"    y = 1, c = 0: {np.mean(split_df.loc[split_df['y'] == 1, 'place'] == 0):.3f}, n = {np.sum((split_df['y'] == 1) & (split_df['place'] == 0))}")
    print(f"    y = 1, c = 1: {np.mean(split_df.loc[split_df['y'] == 1, 'place'] == 1):.3f}, n = {np.sum((split_df['y'] == 1) & (split_df['place'] == 1))}")

### Assign places to train, val, and test set
place_ids_df = pd.read_csv(
    os.path.join(places_dir, 'categories_places365.txt'),
    sep=" ",
    header=None,
    names=['place_name', 'place_id'],
    index_col='place_id')

for idx, target_places in enumerate(target_places):
    place_filenames = []

    for target_place in target_places:
        target_place_full = f'/{target_place[0]}/{target_place}'
        assert (np.sum(place_ids_df['place_name'] == target_place_full) == 1)
        place_id = place_ids_df.index[place_ids_df['place_name'] == target_place_full][0]
        print(f'background slot {idx} {target_place_full} has id {place_id}')

        # Read place filenames associated with target_place
        place_filenames += [
            f'/{target_place[0]}/{target_place}/{filename}' for filename in os.listdir(
                os.path.join(places_dir, 'data_large', target_place[0], target_place))
            if filename.endswith('.jpg')]

    random.shuffle(place_filenames)

    # Assign each filename to an image
    indices = (df.loc[:, 'place'] == idx)
    assert len(place_filenames) >= np.sum(indices),\
        f"Not enough places ({len(place_filenames)}) to fit the dataset ({np.sum(df.loc[:, 'place'] == idx)})"
    df.loc[indices, 'place_filename'] = place_filenames[:np.sum(indices)]

### Write dataset to disk
os.makedirs(output_subfolder, exist_ok=True)

df.to_csv(os.path.join(output_subfolder, 'metadata.csv'))

# 统计处理情况
processed_count = 0
skipped_count = 0
copied_count = 0

processed_details = {
    'processed_count': processed_count,
    'skipped_count': skipped_count,
    'copied_count': copied_count
}

def should_skip_background_replacement(img_filename, place_label, water_bg_annotations):
    """
    判断是否应该跳过背景替换
    
    Args:
        img_filename: 图片文件名
        bird_label: 鸟类标签 (0=陆鸟, 1=水鸟)
        place_label: 背景标签 (0=陆地背景, 1=水域背景)
        water_bg_annotations: 水域背景标注字典
    
    Returns:
        bool: True表示跳过背景替换，False表示需要背景替换
    """
    if place_label == 1 and img_filename in water_bg_annotations:
        return True
    elif place_label == 0 and img_filename not in water_bg_annotations:
        return True
    else:
        return False

for i in tqdm(df.index):
    img_filename = df.loc[i, 'img_filename']
    bird_label = df.loc[i, 'y']
    place_label = df.loc[i, 'place']
    
    # 判断是否需要跳过背景替换
    if should_skip_background_replacement(img_filename, place_label, water_background_annotations):
        # 直接复制原图
        src_path = os.path.join(cub_dir, 'CUB_200_2011', 'images', img_filename)
        output_path = os.path.join(output_subfolder, img_filename)
        os.makedirs('/'.join(output_path.split('/')[:-1]), exist_ok=True)
        shutil.copy2(src_path, output_path)
        copied_count += 1
    else:
        # 执行背景替换
        # Load bird image and segmentation
        img_path = os.path.join(cub_dir, 'CUB_200_2011', 'images', img_filename)
        seg_path = os.path.join(cub_dir, 'segmentations', img_filename.replace('.jpg','.png'))
        img_np = np.asarray(Image.open(img_path).convert('RGB'))
        seg_np = np.asarray(Image.open(seg_path).convert('RGB')) / 255

        # Load place background
        # Skip front /
        place_path = os.path.join(places_dir, 'data_large', df.loc[i, 'place_filename'][1:])
        place = Image.open(place_path).convert('RGB')

        img_black = Image.fromarray(np.around(img_np * seg_np).astype(np.uint8))
        combined_img = combine_and_mask(place, seg_np, img_black)

        output_path = os.path.join(output_subfolder, img_filename)
        os.makedirs('/'.join(output_path.split('/')[:-1]), exist_ok=True)

        combined_img.save(output_path)
        processed_count += 1

print(f"\n处理完成统计:")
print(f"背景替换处理: {processed_count} 张图片")
print(f"直接复制原图: {copied_count} 张图片")
print(f"总计: {processed_count + copied_count} 张图片")
print(f"跳过不必要处理的比例: {copied_count / (processed_count + copied_count) * 100:.2f}%")

processed_details = {
    'processed_count': processed_count,
    'copied_count': copied_count,
    'total': processed_count + copied_count,
    'skip_percentage': float(copied_count / (processed_count + copied_count) * 100) if (processed_count + copied_count) > 0 else 0.0
}

config_summary = {
    'dataset_name': dataset_name,
    'random_seed': random_seed,
    'val_fraction': val_frac,
    'split_group_ratios': {
        split: {f"y{y}_c{c}": float(weight) for (y, c), weight in ratios.items()}
        for split, ratios in split_group_ratios.items()
    },
    'target_places': target_places,
    'statistics': split_statistics,
    'processing': processed_details
}

with open(os.path.join(output_subfolder, 'config.json'), 'w') as f:
    json.dump(config_summary, f, indent=2)
