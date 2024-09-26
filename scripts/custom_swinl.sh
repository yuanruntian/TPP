OUTPUT_DIR=/work_dir/custom_swinl

# training
python3 -m torch.distributed.launch --nproc_per_node=2 --master_port=29700 --use_env \
main.py --with_box_refine --freeze_text_encoder --binary \
--epochs 5 --lr_drop 3 5 \
--lr=1e-5 \
--lr_backbone=5e-6 \
--num_frames=3 \
--output_dir=${OUTPUT_DIR} \
--online \
--backbone swin_l_p4w7 \
--pretrained_weights=/work_dir/pretrained_weights/swin_large_pretrained.pth

# inference
CHECKPOINT=${OUTPUT_DIR}/checkpoint.pth
python3 inference_custom.py --with_box_refine --freeze_text_encoder --binary \
--output_dir=${OUTPUT_DIR} --resume=${CHECKPOINT}  --visualize --draw_init_point \
--online \
--ngpu=2 \
--backbone swin_l_p4w7

python eval_custom.py --path ${OUTPUT_DIR}

echo "Working path is: ${OUTPUT_DIR}"

# CUDA_VISIBLE_DEVICES=0,1 sh scripts/custom_swinl.sh 
