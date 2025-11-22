cd ..
nohup python -u test_vllm_infer.py \
    --editor_name "liveedit" \
    --model_name "llava-v1.5-7b" \
    --data_name "WaterBird" \
    --data_filename "edit_annotations_truelabel_balanced" \
    --split "test" \
    --data_size 2260 \
    --sequential_edit_n 2260 \
    --device "6" \
    --infer_data_name CUB_200_2011 > OUT_infer.log 2>&1 &