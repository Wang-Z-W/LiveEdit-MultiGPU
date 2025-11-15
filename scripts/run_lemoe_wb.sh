cd ..
CUDA_VISIBLE_DEVICES=4,5 python test_vllm_edit.py \
    -dvc "auto" \
    -en "lemoe_vl" \
    -mn "blip2" \
    -dn "WaterBird" \
    -dfn "edit_annotations_truelabel_balanced" \
    -spt "train" \
    -dsn 999999 \
    -sen 999999 \
    -saveckpt