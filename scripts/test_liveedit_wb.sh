cd ..
python -m pdb test_vllm_edit.py \
    -dvc "4,5" \
    -en "liveedit" \
    -mn "llava-v1.5-7b" \
    -dn "WaterBird" \
    -dfn edit_annotations_truelabel \
    -spt test \
    -dsn 9999999 \
    -sen 9999999 \
    -ckpt "records/liveedit/llava-v1.5-7b/WaterBird-2025.11.14-16.37.25/checkpoints/epoch-9-i-20000-ema_loss-0.3013" \
    -saveckpt