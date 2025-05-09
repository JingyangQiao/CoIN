#!/bin/bash

python ./instruct/LoRASelect.py --codebook ./instruct/codebooks/ --instruct ./playground/Instructions/VizWiz/val.json

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

if [ ! -n "$1" ] ;then
    STAGE='CIA_Task8_Type1'
else
    STAGE=$1
fi

if [ ! -n "$2" ] ;then
    MODELPATH='./checkpoints/Instruction/Only_Pretrain_1.5_CIA/OCRVQA/llava-1.5-7b-lora'
else
    MODELPATH=$2
fi

RESULT_DIR="./results/CoIN/VizWiz"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.CoIN.model_vizwiz \
        --model-path $MODELPATH \
        --model-base ./checkpoints/Vicuna/vicuna-7b-v1.5 \
        --question-file ./playground/Instructions/VizWiz/val.json \
        --image-folder ./cl_dataset \
        --answers-file $RESULT_DIR/$STAGE/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &

done

wait

output_file=$RESULT_DIR/$STAGE/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $RESULT_DIR/$STAGE/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python llava/eval/CoIN/eval_vizwiz.py --annotation-file ./playground/Instructions/VizWiz/val.json --result-file $RESULT_DIR/$STAGE/merge.jsonl --output-dir $RESULT_DIR/$STAGE

#python scripts/convert_vizwiz_for_submission.py \
#    --annotation-file ./playground/Instructions/VizWiz/test.json \
#    --result-file $output_file \
#    --result-upload-file $RESULT_DIR/$STAGE/upload.json

#python llava/eval/CoIN/create_prompt.py \
#    --rule llava/eval/CoIN/rule.json \
#    --questions ./playground/Instructions/VizWiz/test.json \
#    --results $output_file \