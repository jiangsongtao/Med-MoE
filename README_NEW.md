<img src="https://github.com/jiangsongtao/TinyMed/assets/43131870/bd7a9801-f94f-485b-9186-8d83620d0bc2" width="400">

**Med-MoE** is a novel and lightweight framework designed to handle both discriminative and generative multimodal medical tasks. It employs a three-step learning process: aligning multimodal medical images with LLM tokens, instruction tuning with a trainable router for expert selection, and domain-specific MoE tuning. Our model stands out by incorporating highly specialized domain-specific experts, significantly reducing the required model parameters by 30%-50% while achieving superior or on-par performance compared to state-of-the-art models. This expert specialization and efficiency make Med-MoE highly suitable for resource-constrained clinical settings.
<img width="515" alt="model" src="https://github.com/jiangsongtao/TinyMed/assets/43131870/21a9246d-698f-492f-ab6f-351cf97b055c">


## Environment Setup

**Prepare the Environment**

1. Clone and navigate to the TinyMed project directory:
   ```bash
   cd TinyMed
   ```

2. Set up your environment:
   ```bash
   conda create -n tinymed python=3.10 -y
   conda activate tinymed
   pip install --upgrade pip
   pip install -e .
   pip install -e ".[train]"
   pip install flash-attn --no-build-isolation
   ```

3. Replace the default MoE with our provided version.

4. Please download the domain-specific router provided by us or trained by yourself, and replace its path in the `moellava/model/language_model/llava_stablelm_moe.py` file.

## Training

**Prepare the Datasets**

   Utilize the LLaVA-Med Datasets for training:
   - **For Pretrained Models**: [LLaVA-Med Alignment Dataset](https://drive.google.com/file/d/1cV_Y30VbMI9R9KcuBd_EiK738kDwcxxA/view?usp=sharing)
   - **For Instruction-Tuning**: [LLaVA-Med Instruct Dataset](https://drive.google.com/file/d/1Dzop-vqsSuieuXFOZHxbkHIfLR9lePa-/view?usp=drive_link)
   - **For Pretrained and SFT Image**:wget https://hanoverprod.z21.web.core.windows.net/med_llava/llava_med_image_urls.jsonl and python download_image.py(Don't forget to replace your path)
   - **For MoE-Tuning Stage**: [Training Jsonl](https://drive.google.com/file/d/1mf3lyW7CbfCowGC58gXsam-3dPwIenbJ/view?usp=sharing)
   - **MoE-Tuning Stage Image Data**: Note that some images from LLaVA-Med are no longer available; these have been excluded from training. [Stage3 ImageData](https://drive.google.com/file/d/1l9hnxa2Y3D8rhNLldtCQ0vGPhsiWH_Su/view?usp=sharing)
   -**Test.json for VQA**:https://drive.google.com/file/d/1pyGsm8G0Gig63DAnOdLuUn3IyxrztWtR/view?usp=sharing
## Web Launch

**Launch the Web Interface**

   Use DeepSpeed to start the Gradio web server:
   - **Phi2 Model**:
     ```bash
     deepspeed --include localhost:0 moellava/serve/gradio_web_server.py --model-path "./MedMoE-phi2"
     ```
   - **StableLM Model**:
     ```bash
     deepspeed --include localhost:0 moellava/serve/gradio_web_server.py --model-path "./MedMoE-stablelm-1.6b"
     ```

## CLI Inference

 **Command Line Inference**
   Execute models from the command line:
   - **Phi2 Model**:
     ```bash
     deepspeed --include localhost:0 moellava/serve/cli.py --model-path "./MedMoE-phi2" --image-file "image.jpg"
     ```
   - **StableLM Model**:
     ```bash
     deepspeed --include localhost:0 moellava/serve/cli.py --model-path "./MedMoE-stablelm-1.6b" --image-file "image.jpg"
     ```

## Model Zoo

**Available Models**
   We will release model weights within two weeks


## Evaluation

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
        --conv-mode stablelm/phi2 &
done

# Combine JSONL results into one file
cat ./test_llava-13b-chunk2_{0..1}.jsonl > ./radvqa.jsonl

# Run evaluation
python run_eval.py \
    --gt ./3vqa/test_rad.json \
    --pred ./radvqa.jsonl \
    --output ./data_RAD/wrong_answers.json
```

## Acknowledgements

Special thanks to these foundational works:
- [MoE-LLaVA](https://github.com/PKU-YuanGroup/MoE-LLaVA)
- [LLaVA-Med](https://github.com/microsoft/LLaVA-Med)
```
