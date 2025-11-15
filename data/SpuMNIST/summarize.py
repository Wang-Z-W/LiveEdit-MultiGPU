import json
import os
import sys
import pyrootutils

root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True)

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

train_root = "spumnist_train"
test_root = "spumnist_test"

def get_spu_dist(anns):
    dist = {}
    for ann in anns:
        label = LABEL2DIGIT[ann['true_label']] if ann.get('true_label', None) else LABEL2DIGIT[ann['pred']]
        spu_attr = 'rectangle' in ann['reg_to_attr']
        if label not in dist.keys():
            dist[label] = [1, int(spu_attr)]
        else:
            dist[label][0] += 1
            dist[label][1] += int(spu_attr)
    return dict(sorted(dist.items(), key=lambda x: x[0], reverse=False))

def _summarize(dir_path):
    for dataset in sorted([d for d in os.listdir(dir_path) if d.endswith(".json")], key=lambda x: (x.split("_")[-1].split(".")[0])):
        train_set = json.load(open(os.path.join(dir_path, dataset)))
        metadata = train_set.get('metadata', {})
        anns = train_set.get('annotations', [])
        spu_dist = get_spu_dist(anns)
        print(f"Dataset: {dataset}, spurious textual attribute: {metadata.get('spurious_textual_attribute', 'N/A')}, Cramer V score: {metadata.get('cramer_v_score', 'N/A')}, data size: {metadata.get('num_images', 'N/A')}")
        for label in spu_dist.keys():
            num, num_spu = spu_dist[label]
            print(f"Label: {label}, num: {num}, num_spu: {num_spu}")


if __name__ == "__main__":
    dir_path = sys.argv[1]
    dir_path = root / "data" / "SpuMNIST" / dir_path
    _summarize(dir_path)