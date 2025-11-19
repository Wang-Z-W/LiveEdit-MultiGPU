from utils import get_full_model_name, load_vllm_editor
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from transformers import LlavaForConditionalGeneration, LlavaProcessor
import argparse
import os
import pandas as pd
from tqdm import tqdm
from PIL import Image
import torch


def get_cfg():
    parser = argparse.ArgumentParser('VLLM inference arguments')
    parser.add_argument('--editor_name', type=str, default='ft_vl', help='Editor name: ft_vl, lemoe_vl, ...', required=True)
    parser.add_argument('--model_name', type=str, default='llava', help='Model name: blip2, llava, ...', required=True)
    parser.add_argument('--data_name', type=str, default=None, help='Data name: waterbird, cub, ...')
    parser.add_argument('--data_filename', type=str, default=None, help='Data file name')
    parser.add_argument('--split', type=str, default=None, help='Split: train, val, test, ...')
    parser.add_argument('--data_size', type=int, default=None, help='Data sample number')
    parser.add_argument('--sequential_edit_n', type=int, default=None, help='Sequential edit number')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--infer_data_name', type=str, default=None, help='Inference data')
    cfg = parser.parse_args()
    return cfg

def infer(editor, text: str, image: Image.Image, max_length: int = 200) -> str:
    model = editor.vllm.model
    processor = editor.vllm.processor
    
    # llm_input_embeds, vt_range = editor.vllm.get_llm_input_embeds([text], [image])
    # llm_output_logits = editor.vllm.get_llm_outpt(llm_input_embeds, vt_range).logits
    # llm_output_ids = torch.softmax(llm_output_logits, -1).argmax(-1)
    
    # assistant_answer = processor.batch_decode(llm_output_ids, skip_special_tokens=True)[0].strip()
    
    model.eval()
    with torch.no_grad():
        if isinstance(model, LlavaForConditionalGeneration):
            Conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text},
                        {"type": "image"},
                    ]
                }
            ]
            prompt = processor.apply_chat_template(Conversation, add_generation_prompt=True)
        elif isinstance(model, Blip2ForConditionalGeneration):
            template = "Question: {} Answer:"
            prompt = template.format(text)
            
        if hasattr(model, "generation_config") and getattr(model.generation_config, "max_length", None) is not None:
            model.generation_config.max_length = None

        inputs = processor(images=image, text=prompt, return_tensors='pt').to(model.device)
        generated_ids = model.generate(**inputs, max_length=max_length, do_sample=False)
        assistant_answer = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        
    return assistant_answer

def infer_dataset(cfg, editor, infer_data, infer_results_save_path: str):
    # Inference
    infer_results = []
    for ann in tqdm(infer_data.data, total=infer_data.__len__(), desc=f'{cfg.model_name} infering on {cfg.infer_data_name}'):
        img_id = ann['img_id']
        img_path = ann['img_path']
        img = ann['image']
        bird_type = ann['bird_type']
        background = ann['background']
        text = ann['text']
        vllm_out = [infer(editor, t, img) for t in text] if isinstance(text, list) else infer(editor, text, img)
        infer_results.append({
            'img_id': img_id,
            'img_path': img_path,
            'bird_type': bird_type,
            'background': background,
            'text': text,
            'vllm_out': vllm_out,
        })
    
    infer_results_df = pd.DataFrame(infer_results)
    infer_results_df.to_csv(infer_results_save_path, index=False)
    print(f"Save inference results to {infer_results_save_path}")

def infer_dataset_single_forward(cfg, editor, infer_data, infer_results_save_path: str):
    vllm = editor.vllm
    def get_eval_input_target(prompt, image, target):
            (x, vt_range), _, m = vllm.prompts_imgs_target_to_xym([prompt], [image], ' ')
            x['query_triple'] = (prompt, image, target)
            x['query_range'] = (0, x['inputs_embeds'].shape[1] - m.shape[1] + 1)
            y = vllm.processor.tokenizer([target], return_tensors='pt', padding=True).input_ids[:,1:].to(vllm.device[-1])
            return (x, vt_range), y
    def accuracy_and_prediction(input_embeds, vt_range, label_ids):
        # label_ids/label_masks: [1, l2]
        assert len(label_ids) == 1
        logits = vllm.get_llm_outpt(input_embeds, vt_range).logits # [1,l1,d]
        pre_y = torch.softmax(logits, -1).argmax(-1) # [1, l1]
        pre_y = pre_y[:, -1:] # [1, l2]
        acc = (pre_y == label_ids).sum() / label_ids.shape[1]
        return float(acc), pre_y
    def infer_single(editor, prompt, image, target):
        (input_embeds, vt_range), label_ids = get_eval_input_target(prompt, image, target)
        acc, pre_y = accuracy_and_prediction(input_embeds, vt_range, label_ids)
        pred = vllm.processor.tokenizer.decode(pre_y[0])
        return acc, pre_y, pred
    
    infer_results = []
    for ann in tqdm(infer_data.data, total=infer_data.__len__(), desc=f'{cfg.model_name} infering on {cfg.infer_data_name}'):
        prompts = ann['text'] if isinstance(ann['text'], list) else [ann['text']]
        image = ann['image']
        target = ann['bird_type']
        vllm_outs = {}
        for p in prompts:
            acc, pre_y, pred = infer_single(editor, p, image, target)
            vllm_outs[p] = {
                'acc': acc,
                'pred_ids': pre_y,
                'pred_text': pred,
            }
        infer_results.append({
            'img_id': ann['img_id'],
            'img_path': ann['img_path'],
            'bird_type': ann['bird_type'],
            'background': ann['background'],
            'text': prompts,
            'vllm_out': vllm_outs,
        })
        import pdb; pdb.set_trace()

    
def infer_single(editor):
    image = Image.open('/data/wzw/LiveEdit/data/CUB_200_2011/CUB_200_2011/images/094.White_breasted_Nuthatch/White_Breasted_Nuthatch_0003_86029.jpg').convert('RGB')
    text = "What type of bird is in the image? Choose from: water bird, land bird"
    vllm_out = infer(editor, text, image)
    import pdb; pdb.set_trace()
    print(vllm_out)
    
def set_device(cfg):
    cfg.device = [f"cuda:{d}" for d in cfg.device.split(',')]

if __name__ == "__main__":
    cfg = get_cfg()
    set_device(cfg)
    cfg.editor_name = cfg.editor_name.lower()
    cfg.model_name = get_full_model_name(cfg.model_name)
    cfg.data_name = cfg.data_name.upper() if cfg.data_name is not None else None
    print(cfg)
    
    proj_dir = os.path.join('eval_results', cfg.editor_name, cfg.model_name,
                             (cfg.data_name if cfg.data_name is not None else ''),
                             (cfg.data_filename if cfg.data_filename is not None else ''),
                             (cfg.split if cfg.split is not None else ''),
                             (f'data_size_{cfg.data_size}' if cfg.data_size is not None else ''),
                             (f'sequential_edit_{cfg.sequential_edit_n}' if cfg.sequential_edit_n is not None else ''))
    if not os.path.exists(proj_dir): os.makedirs(proj_dir)
    cfg.editor_ckpt_path = os.path.join(proj_dir, 'checkpoint_session_0.pt')
    if not os.path.exists(cfg.editor_ckpt_path): cfg.editor_ckpt_path = None
    infer_results_save_path = os.path.join(proj_dir, f'infer_results_{cfg.model_name}_on_{cfg.infer_data_name}.csv')
    
    editor = load_vllm_editor(cfg.editor_name, cfg.model_name, cfg.device, None, cfg.editor_ckpt_path, False)
    editor.vllm._wrap_full_model_forward()
    
    # Load inference data
    if cfg.infer_data_name == 'CUB_200_2011':
        from dataset.vllm import CUBInferenceDataset
        data_path = 'data/CUB_200_2011/CUB_200_2011/metadata.csv'
        img_dir = 'data/CUB_200_2011/CUB_200_2011/images'
        infer_data = CUBInferenceDataset(data_path, img_dir)
    elif cfg.infer_data_name == 'CUB_200_2011_reverse':
        from dataset.vllm import CUBInferenceDataset
        data_path = 'data/CUB_200_2011/CUB_200_2011/metadata_reverse.csv'
        img_dir = 'data/CUB_200_2011/CUB_200_2011/images_reverse'
        infer_data = CUBInferenceDataset(data_path, img_dir)
    else:
        infer_data = None
        
    # Inference
    infer_dataset(cfg, editor, infer_data, infer_results_save_path)

    # infer_dataset_single_forward(cfg, editor, infer_data, infer_results_save_path)