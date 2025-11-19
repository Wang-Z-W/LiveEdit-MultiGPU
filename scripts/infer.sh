cd ..
nohup python -u test_vllm_infer.py \
    --editor_name "liveedit" \
    --model_name "llava-v1.5-7b" \
    --data_name "WaterBird" \
    --data_filename "edit_annotations_truelabel" \
    --split "test" \
    --data_size 5794 \
    --sequential_edit_n 5794 \
    --device "5" \
    --infer_data_name CUB_200_2011 > OUT_infer.log 2>&1 &