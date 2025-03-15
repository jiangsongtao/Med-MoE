Our paper's link is [Med-MoE: Mixture of Domain-Specific Experts for Lightweight Medical Vision-Language Models](https://arxiv.org/pdf/2404.10237.pdf)

<img width="600" alt="model" src="https://github.com/jiangsongtao/TinyMed/assets/43131870/956aa268-1c75-44b3-938f-40fbbd8a53b7">


**Med-MoE** is a novel and lightweight framework designed to handle both discriminative and generative multimodal medical tasks. It employs a three-step learning process: aligning multimodal medical images with LLM tokens, instruction tuning with a trainable router for expert selection, and domain-specific MoE tuning. Our model stands out by incorporating highly specialized domain-specific experts, significantly reducing the required model parameters by 30%-50% while achieving superior or on-par performance compared to state-of-the-art models. This expert specialization and efficiency make Med-MoE highly suitable for resource-constrained clinical settings.

<img width="800" alt="model" src="https://github.com/jiangsongtao/TinyMed/assets/43131870/21a9246d-698f-492f-ab6f-351cf97b055c">


## Environment Setup

**Prepare the Environment**

1. Clone and navigate to the Med-MoE project directory:
   ```bash
   cd Med-MoE
   ```

2. Set up your environment:
  
CUDA 12.4 is recommended. However, you can use another version as long as you update the corresponding package versions in your pymroject.poml file and adjust the PyTorch download lines accordingly.
   ```bash
   conda create -n Med-MoE python=3.10 -y
   conda activate Med-MoE
   pip install --upgrade pip
   pip install -e .
   pip install -e ".[train]"
   pip3 install torch torchvision torchaudio
   pip install flash-attn --no-build-isolation
   ```

3. Please download the domain-specific router provided by us or trained by yourself, and replace its path in the `moellava/model/language_model/llava_stablelm_moe.py` file.

4. Download the corresponding Clip, Phi2, and Stablelm models into the current folder.

## Training

**Prepare the Datasets**

   Utilize the LLaVA-Med Datasets for training:
   - **For Pretrained Models**: [LLaVA-Med Alignment Dataset](https://drive.google.com/file/d/1cV_Y30VbMI9R9KcuBd_EiK738kDwcxxA/view?usp=sharing)
   - **For Instruction-Tuning**: [LLaVA-Med Instruct Dataset](https://drive.google.com/file/d/1Dzop-vqsSuieuXFOZHxbkHIfLR9lePa-/view?usp=drive_link)
   - **For Pretrained and SFT Image**:wget https://hanoverprod.z21.web.core.windows.net/med_llava/llava_med_image_urls.jsonl and python download_image.py(Don't forget to replace your path)
   - **For MoE-Tuning Stage**: [Training Jsonl](https://drive.google.com/file/d/1mf3lyW7CbfCowGC58gXsam-3dPwIenbJ/view?usp=sharing)
   - **MoE-Tuning Stage Image Data**: Note that some images from LLaVA-Med are no longer available; these have been excluded from training. [Stage3 ImageData](https://drive.google.com/file/d/1l9hnxa2Y3D8rhNLldtCQ0vGPhsiWH_Su/view?usp=sharing)
   -**Test.json for VQA**:https://drive.google.com/file/d/1pyGsm8G0Gig63DAnOdLuUn3IyxrztWtR/view?usp=sharing
  
The data can be organized as shown below. You can modify it by editing [download_images.py](./download_images.py), but be sure to update the corresponding content in the scripts and elsewhere accordingly: 

```
Med-MoE
├── moe
├── moellava
├── data
│   ├── 3vqa
│   │   ├── images
│   │   │   ├── data_RAD
│   │   │   ├── pvqa
│   │   │   └── slake
│   │   ├── test_pvqa.json
│   │   ├── test_slake.json
│   │   ├── test_rad.json
│   │   └── train_all.json
│   ├── alignment
│   │   └── llava_med_alignment_500k_filter.json
│   ├── images
│   ├── instruct
│   ├── pmc_articles
│   └── llava_med_image_urls.jsonl
│
...
```

## Train

Sequently run the scripts in `scripts/train_scripts`. The following one is an example for `Phi2`:

```
bash scripts/train_scripts/phi2/pretrain.sh
bash scripts/train_scripts/phi2/finetune.sh
bash scripts/train_scripts/phi2/finetune_moe_allvqa.sh
```
     
## Web Launch

**Launch the Web Interface**

   Use DeepSpeed to start the Gradio web server:
   - **Phi2 Model**:
     ```bash
     deepspeed --include localhost:0 moellava/serve/gradio_web_server.py --model-path "./MedMoE-phi2"
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

- **[Stage1](https://huggingface.co/JsST/TinyMed/tree/main/Stage1)**: stage1 models.
- **[Stage2](https://huggingface.co/JsST/TinyMed/tree/main/Stage2)**: stage2 models.
- **[Stage3](https://huggingface.co/JsST/Med-MoE/tree/main/stage3)**: stage3 models.


## Evaluation

The evaluation process involves running the model on multiple GPUs and combining the results. Modify [eval.sh](./eval.sh) accordingly and execute it with bash.



## Acknowledgements

Special thanks to these foundational works:
- [MoE-LLaVA](https://github.com/PKU-YuanGroup/MoE-LLaVA)
- [LLaVA-Med](https://github.com/microsoft/LLaVA-Med)
```
@misc{jiang2024medmoemixturedomainspecificexperts,
      title={Med-MoE: Mixture of Domain-Specific Experts for Lightweight Medical Vision-Language Models}, 
      author={Songtao Jiang and Tuo Zheng and Yan Zhang and Yeying Jin and Li Yuan and Zuozhu Liu},
      year={2024},
      eprint={2404.10237},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2404.10237}, 
}
