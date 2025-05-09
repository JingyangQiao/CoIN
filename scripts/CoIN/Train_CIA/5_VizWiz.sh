################## VICUNA ##################
PROMPT_VERSION=v1
MODEL_VERSION="vicuna-7b-v1.5"
################## VICUNA ##################

python ./instruct/LoRASelect.py --codebook ./instruct/codebooks/ --instruct ./playground/Instructions/VizWiz/train.json

deepspeed --include localhost:0,1,2,3,4,5,6,7 --master_port 29600 llava/train/train_mem.py \
    --deepspeed ./scripts/zero1.json \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --model_name_or_path ./checkpoints/Vicuna/vicuna-7b-v1.5 \
    --previous_task_model_path ./checkpoints/Instruction/Only_Pretrain_1.5_CIA/GQA/llava-1.5-7b-lora \
    --version $PROMPT_VERSION \
    --data_path ./playground/Instructions/VizWiz/train.json \
    --image_folder ./cl_dataset \
    --vision_tower ./checkpoints/openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/Instruction/Only_Pretrain_1.5_CIA/VizWiz/llava-1.5-7b-lora \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none \
    --dema True