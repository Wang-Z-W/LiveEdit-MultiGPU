cd ..
python test_vllm_edit.py \
    -dvc "2" \
    -en "ft_vl" \
    -mn "llava-v1.5-7b" \
    -dn "WaterBird" \
    -dfn "edit_annotations_truelabel_balanced_question2" \
    -spt "train" \
    -dsn 999999 \
    -sen 999999 \
    -saveckpt

# Required args
# -dvc: device
# -en: editor_name
# -mn: edit_model_name
# -dn: data_name
# -dfn: data_filename
# -spt: split
# -dsn: data_sample_n
# -sen: sequential_edit_n
# -saveckpt: save_checkpoint