"""
Get dataset (annotations) for fine-tuning.
The editing target is the true label, e.g. edit on a digit zero to predict zero.
"""

import os
import sys
import json


def main(file_name):
    file_path = os.path.join("annotations", file_name)
    if os.path.exists(file_path):
        anns = json.load(open(file_path, 'r'))
    else:
        print(f"File {file_path} not found.")
        return
    for ann in anns['annotations']:
        ann['alt'] = ann['pred']
        
    with open(os.path.join("annotations", file_name.replace('.json', '_truelabel.json')), 'w') as f:
        json.dump(anns, f, indent=4)


if __name__ == "__main__":
    file_name = sys.argv[1]
    main(file_name)