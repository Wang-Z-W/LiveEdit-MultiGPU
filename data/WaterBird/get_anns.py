import sys
import os
import json
import pandas as pd
import pyrootutils
import random
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm
from dataset_utils import crop_and_resize, combine_and_mask


root = Path("/data/wzw/LiveEdit-4.43.0")

WATERBIRD_CLASSES = {
    0: "land",
    1: "water"
}

PLACE_CLASSES = {
    0: "land",
    1: "water"
}

SPLIT = {
    0: "train",
    1: "val",
    2: "test"
}

REPHRASE = [
    "Is the bird in the image a water bird or a land bird?",
    # "Does this bird primarily inhabit aquatic environments or terrestrial environments?",
    "Classify the bird as water-adapted or land-adapted.",
    # "What type of habitat does the bird favor: wet areas or dry areas?",
    "Is the bird suited for life in water or on land?",
    # "Does the bird have adaptations for swimming or for perching?",
    # "Categorize the bird based on its common habitat: marine/freshwater or ground-based.",
    # "Is the bird's nature oriented towards lakes and rivers or forests and fields?",
    # "What is the bird's primary ecosystem: aquatic or terrestrial?",
    "Does the bird belong to categories associated with water bodies or land masses?"
]

PLACES = [
    ['f/field/cultivated', 'h/hayfield'],  # Land backgrounds
    ['r/river', 'c/coast']]                # Water backgrounds

PLACES365_PATH = "/data/wzw/group_DRO/Places365/data_large"


def get_loc(data):
    id = random.choice(range(len(data['questions'])))
    return data['questions'][id], data['answers'][id]

def get_m_loc(data):
    m_loc = random.choice(data)
    m_loc_q = m_loc['question']
    m_loc_a = random.choice([a for a in m_loc['answer'] if a != ""])
    m_loc_img = m_loc['image']
    return m_loc_img, m_loc_q, m_loc_a

def replace_bg(image_path, place, output_subfolder):
    img_path = root / "data" / "CUB_200_2011" / "CUB_200_2011" / "images" / image_path
    seg_path = root / "data" / "CUB_200_2011" / "segmentations" / image_path.replace('.jpg', '.png')
    
    place_filename = random.choice(PLACES[place])
    place_dir = os.path.join(PLACES365_PATH, place_filename)
    places = random.choice(os.listdir(place_dir))
    
    img_np = np.asarray(Image.open(img_path).convert('RGB'))
    seg_np = np.asarray(Image.open(seg_path).convert('RGB')) / 255

    # Load place background
    # Skip front /
    place_path = os.path.join(place_dir, places)
    place = Image.open(place_path).convert('RGB')

    img_black = Image.fromarray(np.around(img_np * seg_np).astype(np.uint8))
    combined_img = combine_and_mask(place, seg_np, img_black)

    output_path = os.path.join(output_subfolder, image_path)
    os.makedirs('/'.join(output_path.split('/')[:-1]), exist_ok=True)
    
    combined_img.save(output_path)
    
    return image_path

def main():
    waterbird_file_name = sys.argv[1] if len(sys.argv) > 1 else None
    if waterbird_file_name is None:
        print("Please provide the WaterBird file name as an argument.")
        sys.exit(1)
    waterbird_root = (root / 'data' / 'WaterBird' / waterbird_file_name)
    if os.path.exists(waterbird_root / 'metadata.csv'):
        metadata = pd.read_csv(waterbird_root / "metadata.csv")
        print(f"Loaded metadata from {waterbird_root / 'metadata.csv'}.")
    else:
        print(f"Metadata file not found at {waterbird_root / 'metadata.csv'}. Exiting.")
        sys.exit(1)
    loc_train_data = json.load(open(root / "data" / "easy-edit-mm" / "locality" / "NQ dataset" / "train.json"))
    loc_val_data = json.load(open(root / "data" / "easy-edit-mm" / "locality" / "NQ dataset" / "validation.json"))
    m_loc_data = json.load(open(root / "data" / "easy-edit-mm" / "multimodal_locality" / "OK-VQA dataset" / "okvqa_loc.json"))
    rephrase_img_dir = root / "data" / "WaterBird" / f"rephrase_images_{waterbird_file_name}"
    
    anns = []
    INSTRUCT = "Choose from: land, water."
    for _, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Processing annotations and images"):
        image_path = row['img_filename']
        water = WATERBIRD_CLASSES[row['y']]
        place = PLACE_CLASSES[row['place']]
        split = SPLIT[row['split']]
        place_filename = row['place_filename']
        
        src = f"What type of bird is in the image? {INSTRUCT}"
        pred = water
        # alt = WATERBIRD_CLASSES[1 - row['y']]
        alt = water
        
        rephrase = f"{random.choice(REPHRASE)} {INSTRUCT}"
        
        rephrase_image_path = replace_bg(image_path, 1 - row['place'], rephrase_img_dir)
        
        if split == "train":
            loc_q, loc_a = get_loc(loc_train_data)
        else:
            loc_q, loc_a = get_loc(loc_val_data)
            
        m_loc_img, m_loc_q, m_loc_a = get_m_loc(m_loc_data)
        
        anns.append({
            "image": os.path.join("waterbird_complete95_forest2water2_improved", image_path),
            "water": water,
            "place": place,
            "split": split,
            "place_filename": place_filename,
            "src": src,
            "pred": pred,
            "alt": alt,
            "rephrase": rephrase,
            "image_rephrase": os.path.join(f"rephrase_images_{waterbird_file_name}", rephrase_image_path),
            "loc_q": loc_q,
            "loc_a": loc_a,
            "m_loc_img": m_loc_img,
            "m_loc_q": m_loc_q,
            "m_loc_a": m_loc_a,
        })
        
    with open(root / "data" / "WaterBird" / f"edit_annotations_{waterbird_file_name}_truelabel.json", "w") as f:
        json.dump(anns, f, indent=4)


if __name__ == "__main__":
    np.random.seed(0)
    random.seed(0)
    main()