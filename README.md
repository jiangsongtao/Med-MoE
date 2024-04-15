# TinyMed
![image](https://github.com/jiangsongtao/TinyMed/assets/43131870/4d0e388e-de8b-4fad-a0de-6962c5fe19b8)

**TinyMed** is a cost-efficient Machine Learning Large Language Model (MLLM) developed for the medical domain. Inspired by MoE-LLaVA, this model significantly reduces parameter count while matching or exceeding the capabilities of larger 7B models.
![image](https://github.com/jiangsongtao/TinyMed/assets/43131870/fbe974db-7d2b-47db-ae93-a3a71f9979a5)


## Environment Setup

1. **Prepare the Environment**
   Clone and navigate to the TinyMed project directory, then set up your environment:
   ```bash
   cd TinyMed
   conda create -n tinymed python=3.10 -y
   conda activate tinymed
   pip install --upgrade pip
   pip install -e .
   pip install -e ".[train]"
   pip install flash-attn --no-build-isolation
## Training

2. **Prepare the Datasets**
   Utilize the LLaVA-Med Datasets for training:
   - **For Pretrained Models**: [LLaVA-Med Alignment Dataset](https://drive.google.com/file/d/1cV_Y30VbMI9R9KcuBd_EiK738kDwcxxA/view?usp=drive_link)
   - **For Instruction-Tuning**: [LLaVA-Med Instruct Dataset](https://drive.google.com/file/d/1Dzop-vqsSuieuXFOZHxbkHIfLR9lePa-/view?usp=drive_link)
   - **For MoE-Tuning Stage**: [Training Jsonl](https://drive.google.com/file/d/1Dzop-vqsSuieuXFOZHxbkHIfLR9lePa-/view?usp=drive_link)
   - **Image Data**: Note that some images from LLaVA-Med are no longer available; these have been excluded from training.
  [Stage3 Image](https://drive.google.com/file/d/1Dzop-vqsSuieuXFOZHxbkHIfLR9lePa-/view?usp=drive_link)
## Web Launch

3. **Launch the Web Interface**
   Use DeepSpeed to start the Gradio web server:
   - **Phi2 Model**:
     ```bash
     deepspeed --include localhost:0 moellava/serve/gradio_web_server.py --model-path "./Tinymed-phi2"
     ```
   - **StableLM Model**:
     ```bash
     deepspeed --include localhost:0 moellava/serve/gradio_web_server.py --model-path "./Tinymed-stablelm-1.6b"
     ```

## CLI Inference

4. **Command Line Inference**
   Execute models from the command line:
   - **Phi2 Model**:
     ```bash
     deepspeed --include localhost:0 moellava/serve/cli.py --model-path "./Tinymed-phi2" --image-file "image.jpg"
     ```
   - **StableLM Model**:
     ```bash
     deepspeed --include localhost:0 moellava/serve/cli.py --model-path "./Tinymed-stablelm-1.6b" --image-file "image.jpg"
     ```

## Model Zoo

5. **Available Models**
   | Model                       | Activated Parameters | Link                                                  |
   | --------------------------- | -------------------- | ----------------------------------------------------- |
   | MoE-LLaVA-1.6B×4-Top2       | 2.0B                 | [Tinymed-phi2](https://huggingface.co/JsST/TinyMed/tree/main/Tinymed-phi2)       |
   | MoE-LLaVA-1.8B×4-Top2       | 2.2B                 | [Tinymed-stablelm-1.6b](https://huggingface.co/JsST/TinyMed/tree/main/Tinymed-stablelm-1.6b) |


6.## Evaluation

The evaluation process involves running the model on multiple GPUs and combining the results. Below are the detailed steps and commands:

```bash
# Set the number of chunks and GPUs
CHUNKS=2
GPUS=(0 1)

# Run inference on each GPU
for IDX in {0..1}; do
    GPU_IDX=${GPUS[$IDX]}
    PORT=$((${GPUS[$IDX]} + 29500))
    MASTER_PORT_ENV="MASTER_PORT=$PORT"
    deepspeed --include localhost:$GPU_IDX --master_port $PORT model_vqa_med.py \
        --model-path your_model_path \
        --question-file ./test_rad.json \
        --image-folder ./3vqa/images \
        --answers-file ./test_llava-13b-chunk${CHUNKS}_${IDX}.jsonl \
        --temperature 0 \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --conv-mode simple &
done

# Combine JSONL results into one file
cat ./test_llava-13b-chunk2_{0..1}.jsonl > ./radvqa.jsonl

# Run evaluation
python run_eval.py \
    --gt ./3vqa/test_rad.json \
    --pred ./radvqa.jsonl \
    --output ./data_RAD/wrong_answers.json


7. **Acknowledgements**
   Special thanks to these foundational works:
   - [MoE-LLaVA](https://github.com/PKU-YuanGroup/MoE-LLaVA)
   - [LLaVA-Med](https://github.com/microsoft/LLaVA-Med)
