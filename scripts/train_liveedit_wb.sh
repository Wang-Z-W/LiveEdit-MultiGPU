cd ..
nohup python -u train_vllm_editor.py \
    -dvc "0,1" \
    -edvc "2,3" \
    -en liveedit \
    -mn llava-v1.5-7b \
    -dna WaterBird \
    -dfn edit_annotations_truelabel_balanced \
    -spt train \
    -bs 2 \
    -lkpt None \
    -tnp WaterBird-edit_annotations_truelabel_balanced-train \
    -eps 100 \
    -sci 500 > OUT_train_llava_liveedit_wb_balanced_train_100eps_bs2.log 2>&1 &