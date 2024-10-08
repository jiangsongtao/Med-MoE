#!/bin/bash
VT_VERSION=./clip-vit-large-patch14-336
JSON_FOLDER="./3vqa/train_all.json"
IMAGE_FOLDER="./3vqa/images"

moe_mode="sparse"
num_experts=4
top_k_experts=1##top1+meta expert
use_residual=False
router_aux_loss_coef=0##no need for aux_loss
cd ./MoE-LLaVA
deepspeed --include=localhost:0,1,2,3 moellava/train/train_mem.py \
    --moe_enable True --num_experts ${num_experts} --top_k_experts ${top_k_experts} --capacity_factor 1.5 \
    --moe_mode ${moe_mode} --use_residual ${use_residual} --router_aux_loss_coef ${router_aux_loss_coef} \
    --train_modules fc1 fc2 wg \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path ./MoE-LLaVA/checkpoints/llavaphi-2.7b-finetune-3epoch \
    --version phi \
    --data_path ${JSON_FOLDER} \
    --image_folder ${IMAGE_FOLDER} \
    --image_tower $VT_VERSION \
    --image_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length False \
    --bf16 True \
    --output_dir ./MoE-LLaVA/checkpoints/llavaphi-2.7b-finetune-sft3epoch-moe3vqa-version3 \
    --num_train_epochs 9 \
    --per_device_train_batch_size 8\
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
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

