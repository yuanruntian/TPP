OUTPUT_DIR=/work_dir/custom_r50

# training
python3 -m torch.distributed.launch --nproc_per_node=2 --master_port=29000 --use_env \
main.py --with_box_refine --freeze_text_encoder --binary \
--lr_drop 3 5 \
--lr=1e-4 \
--lr_backbone=5e-6 \
--num_frames=3 \
--output_dir=${OUTPUT_DIR} \
--online \
--epochs 5 \
--pretrained_weights=/work_dir/pretrained_weights/r50_pretrained.pth 


# inference
CHECKPOINT=${OUTPUT_DIR}/checkpoint.pth
python3 inference_custom.py --with_box_refine --freeze_text_encoder --binary \
--output_dir=${OUTPUT_DIR} --resume=${CHECKPOINT} \
--ngpu=2 \
--online \

python eval_custom.py --path ${OUTPUT_DIR}

echo "Working path is: ${OUTPUT_DIR}"

# CUDA_VISIBLE_DEVICES=0,1 sh scripts/custom_r50.sh 
