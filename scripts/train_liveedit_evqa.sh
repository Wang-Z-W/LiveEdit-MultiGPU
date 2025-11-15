cd ..
python train_vllm_editor.py \
    -dvc "0,1" \
    -edvc "2,3" \
    -en liveedit \
    -mn llava-v1.5-7b \
    -dna EVQA \
    -bs 2 \
    -lkpt None \
    -tnp EVQA \
    -eps 10 \
    -sci 500