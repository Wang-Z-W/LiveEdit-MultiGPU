import json
import os
import random
from tqdm import tqdm
from PIL import Image
import numpy as np
import shutil
import pyrootutils

# 设置固定的随机种子
SEED = 0
random.seed(SEED)
np.random.seed(SEED)

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

CODEMAP = {
    "zero": "one",
    "one": "two",
    "two": "three",
    "three": "four",
    "four": "five",
    "five": "six",
    "six": "seven",
    "seven": "eight",
    "eight": "nine",
    "nine": "zero",
}

REPHRASE_TEMPLATES = [
    # 基础识别问题
    "What is the number in the image?",
    "What digit is shown in this image?",
    "Which number does this handwritten digit represent?",
    "Can you identify the numerical value in this image?",
    
    # 描述性问题
    "What number do you see when you look at this handwritten digit?",
    "Looking at this image, what digit can you recognize?",
    "Based on the visual patterns, what number is displayed?",
    "What numerical character is depicted in this handwritten sample?",
    
    # 分析性问题（促进深度思考）
    "By analyzing the stroke patterns and shape, what digit is this?",
    "Considering the curvature and line structure, which number is represented?",
    "What digit would you classify this handwritten character as?",
    "Based on the geometric features of this image, what number is it?",
    
    # 认知推理问题
    "If you were to read this handwritten digit, what number would it be?",
    "What numerical value does this handwritten character convey?",
    "Interpreting the visual information, which digit is shown?",
    "What number is expressed through this handwritten form?",
    
    # 专业术语问题
    "What is the mathematical symbol represented in this image?",
    "Which Arabic numeral is illustrated in this handwritten sample?",
    "What single-digit number is portrayed in this visual representation?",
    "Identify the numerical digit present in this handwritten image."
]

INSTRUCTION = "Only respond with exactly one English word. Do not include punctuation, quotes, spaces, numbers, or any other characters."

def _get_rephrase():
    return f"{random.choice(REPHRASE_TEMPLATES)} {INSTRUCTION}"

# 新增：加载 NQ locality 数据（questions/answers）
def _load_nq_loc(loc_path: str):
    data = json.load(open(loc_path, 'r'))
    questions = data["questions"]
    answers = data["answers"]
    assert len(questions) == len(answers), "NQ questions/answers 数量不一致"
    return questions, answers

# 新增：加载 OK-VQA multimodal locality 列表
def _load_okvqa_mloc(m_loc_path: str):
    m_locs = json.load(open(m_loc_path, 'r'))
    assert isinstance(m_locs, list), "OK-VQA m_loc 需为列表"
    return m_locs

# 新增：按索引采样 NQ 的文本 locality
def _get_loc_by_index(i: int, loc_questions, loc_answers):
    idx = i % len(loc_questions)
    return loc_questions[idx], loc_answers[idx]

# 新增：按索引采样 OK-VQA 的多模态 locality
def _get_m_loc_by_index(i: int, m_locs, okvqa_image_root: str = "data/OK-VQA"):
    idx = i % len(m_locs)
    rec = m_locs[idx]
    m_loc_img = os.path.join(okvqa_image_root, rec["image"]) if okvqa_image_root else rec["image"]
    m_loc_q = rec["question"]
    ans = rec["answer"]
    m_loc_a = random.choice(ans) if isinstance(ans, list) else ans
    return m_loc_img, m_loc_a, m_loc_q

def _get_train_annotations(dataset, spu_text_attr: str, size: int=None,
                          loc_questions=None, loc_answers=None, m_locs=None, okvqa_image_root: str = "data/OK-VQA"):
    metadata = dataset['metadata']
    assert metadata['spurious_textual_attribute'] == spu_text_attr, f"Spurious attribute {spu_text_attr} doesn't match the dataset!"
    anns = dataset['annotations']
    if size is None or size > len(anns):
        size = len(anns)
    random.shuffle(anns)
    anns = anns[:size]
    
    new_anns = []
    for i, ann in tqdm(enumerate(anns), total=size, desc=f"Processing annotations for train set {spu_text_attr}"):
        image_id = ann['image_id']
        image_path = ann['image_filepath']
        text = ann['text']
        target = ann['true_label']
        spu_attr = "rectangle" in ann['reg_to_attr']
        reg_to_attr = ann['reg_to_attr']
        reg_coord = ann['region_coord']
        
        rephrased_question = _get_rephrase()
        rephrase_image = image_path.replace('.png', '_rephrase.png')
        
        # 替换：从 NQ/OK-VQA 采样 loc/m_loc
        assert loc_questions is not None and loc_answers is not None, "loc_questions/loc_answers 未加载"
        assert m_locs is not None, "m_locs 未加载"
        loc, loc_ans = _get_loc_by_index(i, loc_questions, loc_answers)
        m_loc, m_loc_a, m_loc_q = _get_m_loc_by_index(i, m_locs, okvqa_image_root)

        item = {
            'image_id': image_id,
            'image': image_path,
            'spu_attr': spu_attr,
            'reg_to_attr': reg_to_attr,
            'text': text,
            'src': f"What is the number in the image? {INSTRUCTION}",
            'pred': target,
            'alt': CODEMAP[target],
            'rephrase': rephrased_question,
            'rephrase_image': rephrase_image,
            'loc': loc,
            'loc_ans': loc_ans,
            'm_loc': m_loc,
            'm_loc_a': m_loc_a,
            'm_loc_q': m_loc_q,
        }
        new_anns.append(item)
    
    return new_anns


def _get_test_annotations(dataset, size: int=None,
                          loc_questions=None, loc_answers=None, m_locs=None, okvqa_image_root: str = "data/OK-VQA"):
    """
    提取 test 标注，与 _get_train_annotations 类似，
    不需要 spu_text_attr 的检查与传入。
    """
    anns = dataset['annotations']
    if size is None or size > len(anns):
        size = len(anns)
    random.shuffle(anns)
    anns = anns[:size]

    assert loc_questions is not None and loc_answers is not None, "loc_questions/loc_answers 未加载"
    assert m_locs is not None, "m_locs 未加载"

    new_anns = []
    for i, ann in tqdm(enumerate(anns), total=size, desc="Processing annotations for test set"):
        image_id = ann['image_id']
        image_path = ann['image_filepath']
        text = ann['text']
        target = ann['true_label']
        reg_to_attr = ann['reg_to_attr']
        reg_coord = ann['region_coord']

        # 根据 reg_to_attr 判断是否包含矩形这一视觉伪特征
        spu_attr = "rectangle" in reg_to_attr

        # 随机重述问题与对应图片路径
        rephrased_question = _get_rephrase()
        rephrase_image = image_path.replace('.png', '_rephrase.png')

        # 按索引从 NQ/OK-VQA 采样 loc/m_loc
        loc, loc_ans = _get_loc_by_index(i, loc_questions, loc_answers)
        m_loc, m_loc_a, m_loc_q = _get_m_loc_by_index(i, m_locs, okvqa_image_root)

        item = {
            'image_id': image_id,
            'image': image_path,
            'spu_attr': spu_attr,
            'reg_to_attr': reg_to_attr,
            'text': text,
            'src': f"What is the number in the image? {INSTRUCTION}",
            'pred': target,
            'alt': CODEMAP[target],
            'rephrase': rephrased_question,
            'rephrase_image': rephrase_image,
            'loc': loc,
            'loc_ans': loc_ans,
            'm_loc': m_loc,
            'm_loc_a': m_loc_a,
            'm_loc_q': m_loc_q,
        }
        new_anns.append(item)

    return new_anns


def _select_train_set_by_cramer(train_root: str, spu_text_attr: str, target_cramer: float):
    # 遍历 train_root 下的 dataset_*.json，选择与目标 train_Cramer 最接近且 spu_text_attr 匹配的文件
    candidates = []
    for fname in os.listdir(train_root):
        if not (fname.startswith("dataset_") and fname.endswith(".json")):
            continue
        path = os.path.join(train_root, fname)
        try:
            data = json.load(open(path, "r"))
        except Exception:
            continue
        meta = data.get("metadata", {})
        if meta.get("spurious_textual_attribute") != spu_text_attr:
            continue
        cvs = meta.get("cramer_v_score", None)
        if cvs is None:
            continue
        try:
            diff = abs(float(cvs) - float(target_cramer))
        except Exception:
            continue
        candidates.append((diff, path, float(cvs)))

    if not candidates:
        raise FileNotFoundError(
            f"No matching train dataset in '{train_root}' for spurious_textual_attribute='{spu_text_attr}'."
        )

    candidates.sort(key=lambda x: x[0])
    _, best_path, best_cvs = candidates[0]
    best_data = json.load(open(best_path, "r"))
    return best_data, best_path, best_cvs


def get_train_anns(train_root, spu_text_attr, train_cramer, train_size):
    # 依据 spu_text_attr 与 train_cramer 选择合适的训练集 JSON
    train_set, selected_path, selected_cvs = _select_train_set_by_cramer(train_root, spu_text_attr, train_cramer)

    # 使用绝对路径加载 NQ 与 OK-VQA
    nq_loc_path_train = os.path.join(root, "data/MMEdit_data/locality/NQ dataset/train.json")
    okvqa_m_loc_path = os.path.join(root, "data/MMEdit_data/multimodal_locality/OK-VQA dataset/okvqa_loc.json")
    okvqa_image_root = os.path.join(root, "data/OK-VQA")

    if not os.path.exists(nq_loc_path_train):
        raise FileNotFoundError(f"NQ path not found: {nq_loc_path_train}")
    if not os.path.exists(okvqa_m_loc_path):
        raise FileNotFoundError(f"OK-VQA path not found: {okvqa_m_loc_path}")

    # 预加载 NQ 和 OK-VQA 数据
    loc_questions, loc_answers = _load_nq_loc(nq_loc_path_train)
    m_locs = _load_okvqa_mloc(okvqa_m_loc_path)

    train_anns = _get_train_annotations(
        train_set,
        spu_text_attr,
        train_size,
        loc_questions=loc_questions,
        loc_answers=loc_answers,
        m_locs=m_locs,
        okvqa_image_root=okvqa_image_root
    )
    
    meta = {
        "spurious_textual_attribute": spu_text_attr,
        "cramer_v_score": selected_cvs,
        "num_images": train_size,
        "selected_file_path": selected_path,
    }

    return {
        "metadata": meta,
        "annotations": train_anns,
    }


def get_test_anns(test_root, test_size):
    test_set_path = os.path.join(test_root, f"dataset.json")
    test_set = json.load(open(test_set_path, 'r'))

    # 使用绝对路径加载 NQ 与 OK-VQA
    nq_loc_path_test = os.path.join(root, "data/MMEdit_data/locality/NQ dataset/validation.json")
    okvqa_m_loc_path = os.path.join(root, "data/MMEdit_data/multimodal_locality/OK-VQA dataset/okvqa_loc.json")
    okvqa_image_root = os.path.join(root, "data/OK-VQA")

    if not os.path.exists(nq_loc_path_test):
        raise FileNotFoundError(f"NQ path not found: {nq_loc_path_test}")
    if not os.path.exists(okvqa_m_loc_path):
        raise FileNotFoundError(f"OK-VQA path not found: {okvqa_m_loc_path}")

    # 预加载 NQ 和 OK-VQA 数据
    loc_questions, loc_answers = _load_nq_loc(nq_loc_path_test)
    m_locs = _load_okvqa_mloc(okvqa_m_loc_path)

    test_anns = _get_test_annotations(
        test_set,
        test_size,
        loc_questions=loc_questions,
        loc_answers=loc_answers,
        m_locs=m_locs,
        okvqa_image_root=okvqa_image_root
    )

    meta = {
        "num_images": test_size,
        "selected_file_path": test_set_path,
    }
    
    return {
        "metadata": meta,
        "annotations": test_anns,
    }


def main(train_root, test_root, train_cramer, train_size, test_size):
    anns_dir = root / "data" / "SpuMNIST" / "annotations"
    if os.path.exists(anns_dir) and os.listdir(anns_dir):
        shutil.rmtree(anns_dir)
    anns_dir.mkdir()
    
    for spu_text_attr in LABEL2DIGIT.keys():
        ann = get_train_anns(train_root, spu_text_attr, train_cramer, train_size)
        with open(anns_dir / f"train_{LABEL2DIGIT[spu_text_attr]}.json", "w") as f:
            json.dump(ann, f, indent=4)
        
    ann = get_test_anns(test_root, test_size)
    with open(anns_dir / f"test.json", "w") as f:
        json.dump(ann, f, indent=4)


if __name__ == "__main__":
    train_root = "spumnist_train"
    test_root = "spumnist_test"
    
    train_size = 8000
    test_size = 2000

    # 提供一个目标 train_Cramer（例如 0.05）
    train_Cramer = 0.8
    
    main(train_root, test_root, train_Cramer, train_size, test_size)
