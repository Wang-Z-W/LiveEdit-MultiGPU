cd ..
python -m pdb test_vllm_edit.py \
    -dvc "4,5" \
    -en "liveedit" \
    -mn "llava-v1.5-7b" \
    -dn "WaterBird" \
    -dfn edit_annotations_truelabel \
    -spt test \
    -dsn 10 \
    -sen 9999999 \
    -ckpt "records/liveedit/llava-v1.5-7b/WaterBird-2025.11.15-13.32.14/checkpoints/epoch-38-i-90000-ema_loss-0.1448" \
    -saveckpt