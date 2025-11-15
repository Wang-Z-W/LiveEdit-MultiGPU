import json
import os


LABEL2DIGIT = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
}

def filter_anns_for_class(file_path, cls):
    assert cls in LABEL2DIGIT.keys(), f"Class label {cls} not in {LABEL2DIGIT.keys()}"
    
    with open(file_path, 'r') as f:
        anns = json.load(f)
    
    metadata = anns['metadata']
    metadata['original_num_images'] = metadata.pop('num_images')
    metadata['filtered_file_path'] = file_path
    metadata['filtered_class'] = cls
    
    new_annotations = [item for item in anns['annotations'] if item['pred']==cls]
    metadata['num_images'] = len(new_annotations)
    
    new_anns = {
        'metadata': metadata,
        'annotations': new_annotations,
    }
    
    new_file_path = file_path.replace('.json', f'_filtered_{LABEL2DIGIT[cls]}.json')
    with open(new_file_path, 'w') as f:
        json.dump(new_anns, f, indent=4)
    
    
if __name__ == '__main__':
    file_path = 'annotations/test.json'
    cls = 'five'
    filter_anns_for_class(file_path, cls)
