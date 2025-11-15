cd ..
python -m pdb test_vllm_edit.py \
    -dvc "0" \
    -en "liveedit" \
    -mn "llava-v1.5-7b" \
    -dn "EVQA" \
    -sen 1000 \
    -ckpt "records/liveedit/llava-v1.5-7b/EVQA-2025.11.13-17.33.47/checkpoints/epoch-9-i-25500-ema_loss-0.0780" \
    -saveckpt