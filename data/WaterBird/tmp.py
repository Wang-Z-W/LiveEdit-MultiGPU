import json


instruction = "Please answer with either water or land, and do not include any additional punctuation, numbers, or characters."

anns = json.load(open("edit_annotations_truelabel_balanced.json"))
for ann in anns:
    src = f"Look at this bird image. Is this a water bird or a land bird? Please answer with 'water' or 'land'."
    ann['src'] = src

with open("edit_annotations_truelabel_balanced_question2.json", "w") as f:
    json.dump(anns, f, indent=4)