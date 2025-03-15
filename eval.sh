# Set the number of chunks and GPUs
CHUNKS=2
GPUS=(0 1)
VT_VERSION=./clip-vit-large-patch14-336

# Run inference on each GPU
for IDX in {0..1}; do
    GPU_IDX=${GPUS[$IDX]}
    PORT=$((${GPUS[$IDX]} + 29500))
    MASTER_PORT_ENV="MASTER_PORT=$PORT"
    deepspeed --include localhost:$GPU_IDX --master_port $PORT model_vqa_med.py \
        --model-path your_model_path \
        --question-file ./data/3vqa/test_rad.json \
        --image-folder ./data/3vqa/images \
        --answers-file ./test_llava-13b-chunk${CHUNKS}_${IDX}.jsonl \
        --image_tower $VT_VERSION \
        --temperature 0 \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --conv-mode phi &
        # --conv-mode stablelm &
done

wait

# Combine JSONL results into one file
cat ./test_llava-13b-chunk2_{0..1}.jsonl > ./radvqa.jsonl

# Run evaluation
python run_eval.py \
    --gt ./data/3vqa/test_rad.json \
    --pred ./radvqa.jsonl \
    --output ./wrong_answers.json