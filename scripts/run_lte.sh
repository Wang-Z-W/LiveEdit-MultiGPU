cd ..
CUDA_VISIBLE_DEVICES=4,5 python train_vllm_editor.py \
    -en lte_vl \
    -mn blip2 \
    -dna EVQA \
    -bs 1 \
    -dvc "cuda:0" \
    -edvc 1 \
    -lkpt None \
    -tnp EVQA \
    -eps 50 \
    -sci 500