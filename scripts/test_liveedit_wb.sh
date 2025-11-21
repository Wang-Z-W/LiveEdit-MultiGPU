cd ..
python -m pdb test_vllm_edit.py \
    -dvc "4,5" \
    -en "liveedit" \
    -mn "llava-v1.5-7b" \
    -dn "WaterBird" \
    -dfn edit_annotations_truelabel_balanced \
    -spt test \
    -dsn 9999999 \
    -sen 9999999 \
    -ckpt "records/liveedit/llava-v1.5-7b/WaterBird-2025.11.19-15.14.49/checkpoints/epoch-42-i-54000-ema_loss-0.0906" \
    -saveckpt