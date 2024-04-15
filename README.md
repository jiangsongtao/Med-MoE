# TinyMed

**TinyMed** is a cost-efficient Machine Learning Large Language Model (MLLM) developed for the medical domain. Inspired by MoE-LLaVA, this model significantly reduces parameter count while matching or exceeding the capabilities of larger 7B models.

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
   - **For MoE-Tuning Stage**: [Training JSONL](https://drive.google.com/file/d/1Dzop-vqsSuieuXFOZHxbkHIfLR9lePa-/view?usp=drive_link)
   - **Image Data**: Note that some images from LLaVA-Med are no longer available; these have been excluded from training.

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

## Related Projects

6. **Acknowledgements**
   Special thanks to these foundational works:
   - [MoE-LLaVA](https://github.com/PKU-YuanGroup/MoE-LLaVA)
   - [LLaVA-Med](https://github.com/microsoft/LLaVA-Med)
