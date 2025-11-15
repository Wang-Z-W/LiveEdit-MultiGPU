import json
import random
from collections import defaultdict, Counter


def analyze_dataset_distribution(data_file):
    """Analyze the distribution of water/land birds and their backgrounds"""
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    # Count distributions
    bird_type_count = Counter()
    background_count = Counter()
    combination_count = defaultdict(lambda: defaultdict(int))
    split_count = defaultdict(lambda: defaultdict(int))
    
    for item in data:
        bird_type = item['water']  # 'water' or 'land'
        background = item['place']  # 'water' or 'land'
        split = item['split']  # 'train', 'val', 'test'
        
        bird_type_count[bird_type] += 1
        background_count[background] += 1
        combination_count[bird_type][background] += 1
        split_count[split][bird_type] += 1
    
    total_samples = len(data)
    
    print("=== Dataset Distribution Analysis ===")
    print(f"Total samples: {total_samples}")
    print(f"\nBird type distribution:")
    for bird_type, count in bird_type_count.items():
        percentage = count / total_samples * 100
        print(f"  {bird_type.capitalize()} birds: {count} ({percentage:.1f}%)")
    
    print(f"\nBackground distribution:")
    for bg_type, count in background_count.items():
        percentage = count / total_samples * 100
        print(f"  {bg_type.capitalize()} background: {count} ({percentage:.1f}%)")
    
    print(f"\nBird type vs Background combination:")
    for bird_type in ['water', 'land']:
        print(f"  {bird_type.capitalize()} birds:")
        for bg_type in ['water', 'land']:
            count = combination_count[bird_type][bg_type]
            bird_total = bird_type_count[bird_type]
            percentage = count / bird_total * 100 if bird_total > 0 else 0
            print(f"    on {bg_type} background: {count} ({percentage:.1f}%)")
    
    print(f"\nSplit distribution:")
    for split in ['train', 'val', 'test']:
        print(f"  {split.capitalize()}:")
        for bird_type in ['water', 'land']:
            count = split_count[split][bird_type]
            print(f"    {bird_type} birds: {count}")
    
    return data, bird_type_count, combination_count, split_count


def create_balanced_dataset(data, target_samples_per_class=None, prefer_matching_background=True):
    """
    Create a balanced dataset with equal numbers of water and land birds
    
    Args:
        data: Original dataset
        target_samples_per_class: Target number of samples per class (water/land)
        prefer_matching_background: If True, prioritize water birds on water background 
                                  and land birds on land background
    """
    # Separate data by bird type and background
    water_birds_water_bg = []
    water_birds_land_bg = []
    land_birds_water_bg = []
    land_birds_land_bg = []
    
    for item in data:
        bird_type = item['water']
        background = item['place']
        
        if bird_type == 'water' and background == 'water':
            water_birds_water_bg.append(item)
        elif bird_type == 'water' and background == 'land':
            water_birds_land_bg.append(item)
        elif bird_type == 'land' and background == 'water':
            land_birds_water_bg.append(item)
        elif bird_type == 'land' and background == 'land':
            land_birds_land_bg.append(item)
    
    print(f"\n=== Available samples by category ===")
    print(f"Water birds on water background: {len(water_birds_water_bg)}")
    print(f"Water birds on land background: {len(water_birds_land_bg)}")
    print(f"Land birds on water background: {len(land_birds_water_bg)}")
    print(f"Land birds on land background: {len(land_birds_land_bg)}")
    
    # Determine target samples per class
    if target_samples_per_class is None:
        # Use the minimum available for balanced sampling
        total_water_birds = len(water_birds_water_bg) + len(water_birds_land_bg)
        total_land_birds = len(land_birds_water_bg) + len(land_birds_land_bg)
        target_samples_per_class = min(total_water_birds, total_land_birds)
    
    print(f"\nTarget samples per class: {target_samples_per_class}")
    
    # Sample water birds (prefer water background)
    selected_water_birds = []
    if prefer_matching_background:
        # First, take as many water birds on water background as possible
        water_on_water = min(len(water_birds_water_bg), target_samples_per_class)
        selected_water_birds.extend(random.sample(water_birds_water_bg, water_on_water))
        
        # Fill remaining with water birds on land background if needed
        remaining_water = target_samples_per_class - water_on_water
        if remaining_water > 0 and len(water_birds_land_bg) > 0:
            water_on_land = min(len(water_birds_land_bg), remaining_water)
            selected_water_birds.extend(random.sample(water_birds_land_bg, water_on_land))
    else:
        # Random sampling from all water birds
        all_water_birds = water_birds_water_bg + water_birds_land_bg
        selected_water_birds = random.sample(all_water_birds, 
                                           min(len(all_water_birds), target_samples_per_class))
    
    # Sample land birds (prefer land background)
    selected_land_birds = []
    if prefer_matching_background:
        # First, take as many land birds on land background as possible
        land_on_land = min(len(land_birds_land_bg), target_samples_per_class)
        selected_land_birds.extend(random.sample(land_birds_land_bg, land_on_land))
        
        # Fill remaining with land birds on water background if needed
        remaining_land = target_samples_per_class - land_on_land
        if remaining_land > 0 and len(land_birds_water_bg) > 0:
            land_on_water = min(len(land_birds_water_bg), remaining_land)
            selected_land_birds.extend(random.sample(land_birds_water_bg, land_on_water))
    else:
        # Random sampling from all land birds
        all_land_birds = land_birds_water_bg + land_birds_land_bg
        selected_land_birds = random.sample(all_land_birds, 
                                          min(len(all_land_birds), target_samples_per_class))
    
    # Combine selected samples
    balanced_dataset = selected_water_birds + selected_land_birds
    random.shuffle(balanced_dataset)  # Shuffle the final dataset
    
    # Analyze the balanced dataset
    print(f"\n=== Balanced Dataset Statistics ===")
    print(f"Total samples: {len(balanced_dataset)}")
    
    balanced_stats = defaultdict(lambda: defaultdict(int))
    split_stats = defaultdict(lambda: defaultdict(int))
    
    for item in balanced_dataset:
        bird_type = item['water']
        background = item['place']
        split = item['split']
        balanced_stats[bird_type][background] += 1
        split_stats[split][bird_type] += 1
    
    for bird_type in ['water', 'land']:
        total_bird = sum(balanced_stats[bird_type].values())
        print(f"\n{bird_type.capitalize()} birds: {total_bird}")
        for bg_type in ['water', 'land']:
            count = balanced_stats[bird_type][bg_type]
            percentage = count / total_bird * 100 if total_bird > 0 else 0
            print(f"  on {bg_type} background: {count} ({percentage:.1f}%)")
    
    print(f"\nSplit distribution in balanced dataset:")
    for split in ['train', 'val', 'test']:
        total_split = sum(split_stats[split].values())
        if total_split > 0:
            print(f"  {split.capitalize()}: {total_split}")
            for bird_type in ['water', 'land']:
                count = split_stats[split][bird_type]
                percentage = count / total_split * 100 if total_split > 0 else 0
                print(f"    {bird_type} birds: {count} ({percentage:.1f}%)")
    
    return balanced_dataset


def save_balanced_dataset(balanced_data, output_file):
    """Save the balanced dataset to a JSON file"""
    with open(output_file, 'w') as f:
        json.dump(balanced_data, f, indent=2)
    print(f"\nBalanced dataset saved to: {output_file}")


def main():
    # Set random seed for reproducibility
    random.seed(42)
    
    input_file = 'edit_annotations.json'
    output_file = 'edit_annotations_truelabel_balanced.json'
    
    print("Analyzing original dataset...")
    data, bird_type_count, combination_count, split_count = analyze_dataset_distribution(input_file)
    
    print("\nCreating balanced dataset...")
    # Create balanced dataset with preference for matching backgrounds
    balanced_data = create_balanced_dataset(
        data, 
        target_samples_per_class=None,  # Will use minimum available
        prefer_matching_background=True
    )
    
    print(f"\nSaving balanced dataset...")
    save_balanced_dataset(balanced_data, output_file)
    
    print(f"\n=== Summary ===")
    print(f"Original dataset: {len(data)} samples")
    print(f"Balanced dataset: {len(balanced_data)} samples")
    print(f"Water birds: ~21% -> 50%")
    print(f"Land birds: ~79% -> 50%")
    print(f"Preference: Water birds mainly on water background, Land birds mainly on land background")


if __name__ == "__main__":
    main()