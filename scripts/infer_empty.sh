cd ..
python test_vllm_infer.py \
    --editor_name "empty" \
    --model_name "llava-v1.5-7b" \
    --infer_data_name CUB_200_2011 \
    --device "cuda:0"