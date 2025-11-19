import json
import os
import pandas as pd


def summarize(anns):
    count = {
        'train': {'water': {'water': 0, 'land': 0}, 'land': {'water': 0, 'land': 0}},
        'test': {'water': {'water': 0, 'land': 0}, 'land': {'water': 0, 'land': 0}},
        'val': {'water': {'water': 0, 'land': 0}, 'land': {'water': 0, 'land': 0}},
        'total': {'water': {'water': 0, 'land': 0}, 'land': {'water': 0, 'land': 0}}
    }
    for ann in anns:
        count[ann['split']][ann['water']][ann['place']] += 1
        count['total'][ann['water']][ann['place']] += 1
    
    print(pd.DataFrame(count))
    print("\n")


if __name__ == "__main__":
    for file_name in os.listdir('./'):
        if file_name.endswith('.json'):
            print(file_name)
            anns = json.load(open(file_name, 'r'))
            summarize(anns)