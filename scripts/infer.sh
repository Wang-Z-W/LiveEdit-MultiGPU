cd ..
python test_vllm_infer.py \
    --editor_name "ft_vl" \
    --model_name "llava-v1.5-7b" \
    --data_name "WaterBird" \
    --data_filename "edit_annotations_truelabel_balanced_question1" \
    --split "train" \
    --data_size 2589 \
    --sequential_edit_n 2589 \
    --device "cuda:1" \
    --infer_data_name CUB_200_2011