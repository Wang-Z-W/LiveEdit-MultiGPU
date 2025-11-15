cd ..
CUDA_VISIBLE_DEVICES=7 python test_vllm_edit.py \
    -en "ft_vl" \
    -mn "llava-v1.5-7b" \
    -sen 99999\
    -dvc "cuda:0" \
    -dn "SpuMNIST" \
    -dfn "train_5_truelabel" \
    -ckpt "None"