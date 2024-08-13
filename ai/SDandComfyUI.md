#  SD 与 ComfyUI

## 0. 模型下载

> stable-diffusion

```bash
https://huggingface.co/runwayml/stable-diffusion-v1-5
https://hf-mirror.com/runwayml/stable-diffusion-v1-5/blob/main/v1-5-pruned-emaonly.ckpt
```

## 1. stable-diffusion-webui

```bash
// Clone the repository
git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git

// Models
models/Stable-diffusion/*.ckpt

// Run
cd stable-diffusion-webui && python webui.py
```

## 2. ComfyUI

### 2.1 安装

```bash
// source
https://www.comfy.org

// Clone the repository
git clone git@github.com:comfyanonymous/ComfyUI.git

// Create an environment with Conda
conda create -n comfyenv
conda activate comfyenv

// Install GPU Dependencies
conda install pytorch-nightly::pytorch torchvision torchaudio -c pytorch-nightly

// Install pip requirements
cd ComfyUI
pip install -r requirements.txt

// Models
models/checkpoints/*.ckpt

// Start the application
cd ComfyUI
python main.py
```

### 2.2 管理器

https://github.com/ltdrdata/ComfyUI-Manager

### 2.3 界面汉化

https://github.com/AIGODLIKE/AIGODLIKE-COMFYUI-TRANSLATION

## 3. civitai

> https://civitai.com

### 3.1 使用

* checkpoint
* LoRA
* workflow

### 3.2 贡献

> Civitai LoRA Trainer

https://education.civitai.com/using-civitai-the-on-site-lora-trainer
https://education.civitai.com/lora-training-glossary

##### step 1 选择类型

* Character: A specific person or character, realistic or anime
* Style: A time period, art style, or general look and feel
* Concept: Objects, clothing, anatomy, poses, etc.

##### step 2 上传数据集

##### step 3 选择底模，选择训练参数

##### step 4 提交、等待、测试

## 4. 工作流

### OpenArt

https://openart.ai/workflows

### liblibAI

https://www.liblib.art/workflows

## 5. 项目

### LivePortrait [ 面部表情复刻 ]

https://huggingface.co/KwaiVGI/LivePortrait

### MimicMotion [ 动作复刻 ]

https://github.com/Tencent/MimicMotion

### CosyVoice [ 声音复刻 ]

https://github.com/FunAudioLLM/CosyVoice

### AnimateDiff [ 转绘动画风格视频 ]

https://huggingface.co/ByteDance/AnimateDiff-Lightning