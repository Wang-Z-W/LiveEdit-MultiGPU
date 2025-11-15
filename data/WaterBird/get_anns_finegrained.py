import argparse
import os
import json
import sys
import random


def filter_fine_grained(anns, bird, background, size=None):
    """
    Filter annotations to only include those with the specified bird type on the specified background type.
    """
    finegrained_anns = []
    random.shuffle(anns)
    for ann in anns:
        if ann["water"] == bird and ann["place"] == background:
            finegrained_anns.append(ann)
        if size is not None and len(finegrained_anns) >= size:
            break
    return finegrained_anns


if __name__ == "__main__":
    random.seed(0)
    parser = argparse.ArgumentParser(description="Get fine-grained annotations for WaterBird dataset")
    parser.add_argument("--bird", type=str, default="water", help="Bird type (water or land)")
    parser.add_argument("--background", type=str, default="water", help="Background type (water or land)")
    parser.add_argument("--size", type=int, default=None, help="Size of the dataset (small or large)")
    args = parser.parse_args()
    
    base_anns = json.load(open("edit_annotations_truelabel.json", "r"))
    finegrained_anns = filter_fine_grained(base_anns, args.bird, args.background, args.size)
    with open(f"edit_annotations_truelabel_finegrained_{args.bird}_on_{args.background}.json", "w") as f:
        json.dump(finegrained_anns, f, indent=4)