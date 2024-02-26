# export MODEL_NAME="playgroundai/playground-v2.5-1024px-aesthetic"
# export OUTPUT_DIR="/root/pokemon"
# export HUB_MODEL_ID="pokemon-lora"
# export DATASET_NAME="lambdalabs/pokemon-blip-captions"

# # accelerate launch  train_text_to_image_lora_sdxl.py \
# # accelerate launch --mixed_precision="fp16"  train_text_to_image_lora_sdxl.py \
# #   --pretrained_model_name_or_path=$MODEL_NAME \
# #   --dataset_name=$DATASET_NAME \
# #   --dataloader_num_workers=8 \
# #   --resolution=512 \
# #   --center_crop \
# #   --random_flip \
# #   --train_batch_size=1 \
# #   --gradient_accumulation_steps=4 \
# #   --max_train_steps=15000 \
# #   --learning_rate=1e-04 \
# #   --max_grad_norm=1 \
# #   --lr_scheduler="cosine" \
# #   --lr_warmup_steps=0 \
# #   --output_dir=${OUTPUT_DIR} \
# #   --push_to_hub \
# #   --hub_model_id=${HUB_MODEL_ID} \
# #   --report_to=wandb \
# #   --checkpointing_steps=500 \
# #   --validation_prompt="A pokemon with blue eyes." \
# #   --validation_epochs=1 \
# #   --seed=1337

#   # NOTE: run validations every epoch
# accelerate launch train_text_to_image_lora_sdxl.py \
#     --pretrained_model_name_or_path $MODEL_NAME  \
#     --pretrained_vae_model_name_or_path=madebyollin/sdxl-vae-fp16-fix \
#     --learning_rate=2e-4 \
#     --rank 128 \
#     --num_train_epochs=150 \
#     --validation_epochs=1 \
#     --checkpointing_steps=100 \
#     --mixed_precision='fp16' \
#     --train_batch_size=2 \
#     --num_validation_images 4 \
#     --dataset_name=$DATASET_NAME \
#     --image_column=image \
#     --caption_column=text \
#     --report_to=wandb \
#     --dataloader_num_workers=16 \
#     --validation_prompt='A pokemon with blue eyes.' \
#     --resume_from_checkpoint=latest \
#     --output_dir ./output/sd-model-finetuned-lora


#!/bin/bash

export PYTHONPATH="/root/playground/diffusers/src:$PYTHONPATH"

export PROGRAM=(
    train_text_to_image_lora_sdxl.py
    --pretrained_model_name_or_path "playgroundai/playground-v2.5-1024px-aesthetic"
    #--pretrained_vae_model_name_or_path=madebyollin/sdxl-vae-fp16-fix
    --learning_rate=2e-4
    --rank 128
    --num_train_epochs=150
    --validation_epochs=10
    --checkpointing_steps=100
    # --mixed_precision='fp16'
    --train_batch_size=2
    --num_validation_images 4
    --train_data_dir='./sculpture'
    --image_column=image
    --caption_column=text
    --report_to=wandb
    # --lr_scheduler=constant_with_warmup
    --dataloader_num_workers=4
    --validation_prompt='sculpture style, a cute bunny'
    --output_dir ./output/sd-model-finetuned-lora
)

    # --resume_from_checkpoint=latest \

accelerate launch "${PROGRAM[@]}" 
