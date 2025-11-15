import os
import json
import pandas as pd
import pyrootutils
import random
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm

root = Path("/data/wzw/LiveEdit-4.43.0")
waterbird_root = root / "data" / "WaterBird" / "waterbird_complete95_forest2water2_improved"

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

def crop_and_resize(source_img, target_img):
    """
    Make source_img exactly the same as target_img by expanding/shrinking and
    cropping appropriately.

    If source_img's dimensions are strictly greater than or equal to the
    corresponding target img dimensions, we crop left/right or top/bottom
    depending on aspect ratio, then shrink down.

    If any of source img's dimensions are smaller than target img's dimensions,
    we expand the source img and then crop accordingly

    Modified from
    https://stackoverflow.com/questions/4744372/reducing-the-width-height-of-an-image-to-fit-a-given-aspect-ratio-how-python
    """
    source_width = source_img.size[0]
    source_height = source_img.size[1]

    target_width = target_img.size[0]
    target_height = target_img.size[1]

    # Check if source does not completely cover target
    if (source_width < target_width) or (source_height < target_height):
        # Try matching width
        width_resize = (target_width, int((target_width / source_width) * source_height))
        if (width_resize[0] >= target_width) and (width_resize[1] >= target_height):
            source_resized = source_img.resize(width_resize, Image.Resampling.LANCZOS)
        else:
            height_resize = (int((target_height / source_height) * source_width), target_height)
            assert (height_resize[0] >= target_width) and (height_resize[1] >= target_height)
            source_resized = source_img.resize(height_resize, Image.Resampling.LANCZOS)
        # Rerun the cropping
        return crop_and_resize(source_resized, target_img)

    source_aspect = source_width / source_height
    target_aspect = target_width / target_height

    if source_aspect > target_aspect:
        # Crop left/right
        new_source_width = int(target_aspect * source_height)
        offset = (source_width - new_source_width) // 2
        resize = (offset, 0, source_width - offset, source_height)
    else:
        # Crop top/bottom
        new_source_height = int(source_width / target_aspect)
        offset = (source_height - new_source_height) // 2
        resize = (0, offset, source_width, source_height - offset)

    source_resized = source_img.crop(resize).resize((target_width, target_height), Image.Resampling.LANCZOS)
    return source_resized

def combine_and_mask(img_new, mask, img_black):
    """
    Combine img_new, mask, and image_black based on the mask

    img_new: new (unmasked image)
    mask: binary mask of bird image
    img_black: already-masked bird image (bird only)
    """
    # Warp new img to match black img
    img_resized = crop_and_resize(img_new, img_black)
    img_resized_np = np.asarray(img_resized)

    # Mask new img
    img_masked_np = np.around(img_resized_np * (1 - mask)).astype(np.uint8)

    # Combine
    img_combined_np = np.asarray(img_black) + img_masked_np
    img_combined = Image.fromarray(img_combined_np)

    return img_combined

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
    metadata = pd.read_csv(waterbird_root / "metadata.csv")
    loc_train_data = json.load(open(root / "data" / "easy-edit-mm" / "locality" / "NQ dataset" / "train.json"))
    loc_val_data = json.load(open(root / "data" / "easy-edit-mm" / "locality" / "NQ dataset" / "validation.json"))
    m_loc_data = json.load(open(root / "data" / "easy-edit-mm" / "multimodal_locality" / "OK-VQA dataset" / "okvqa_loc.json"))
    rephrase_img_dir = root / "data" / "WaterBird" / "rephrase_images"
    
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
            "image_rephrase": os.path.join("rephrase_images", rephrase_image_path),
            "loc_q": loc_q,
            "loc_a": loc_a,
            "m_loc_img": m_loc_img,
            "m_loc_q": m_loc_q,
            "m_loc_a": m_loc_a,
        })
        
    with open(root / "data" / "WaterBird" / "edit_annotations.json", "w") as f:
        json.dump(anns, f, indent=4)


if __name__ == "__main__":
    np.random.seed(0)
    random.seed(0)
    main()