#!/bin/bash
VT_VERSION=./clip-vit-large-patch14-336
JSON_FOLDER="./data/instruct/llava_med_instruct_60k_inline_mention_filter.json"
IMAGE_FOLDER="./data/images"
cd ./MoE-LLaVA
deepspeed moellava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
     --model_name_or_path ./phi-2\
    --version phi \
    --data_path ${JSON_FOLDER}\
    --image_folder ${IMAGE_FOLDER} \
   --image_tower $VT_VERSION \
    --image_projector_type mlp2x_gelu \
    --pretrain_mm_mlp_adapter ./checkpoints/llavaphi-2.7b-pretrain/mm_projector.bin \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/llavaphi-2.7b-finetune-3epoch \
    --num_train_epochs 3 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 300 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --cache_dir "./cache_dir"

