# ViLaMAS

## Environment

The experiments use 4 × NVIDIA H100 80 GB GPUs. The measured NVIDIA
driver version is 580.95.05. 

```bash
conda create -n vilamas python=3.10.20 -y
conda activate vilamas
python -m pip install torch==2.11.0 torchvision==0.26.0 \
  --index-url https://download.pytorch.org/whl/cu130
python -m pip install transformers==4.57.6 accelerate==1.14.0 \
  'qwen-vl-utils[decord]==0.0.14' av==17.1.0 \
  numpy==2.2.6 pillow==12.3.0 pandas==2.3.3 pyarrow==25.0.0 \
  'huggingface-hub[cli]==0.36.2' safetensors==0.8.0 \
  pyyaml psutil==7.2.2 nvidia-ml-py==13.610.43
python -c "import torch; from transformers import Qwen3VLForConditionalGeneration; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

The evaluation backend uses Hugging Face Transformers; vLLM and FlashAttention
are not required. [PyTorch installation reference](https://pytorch.org/get-started/previous-versions/).

## Model downloads

Download the backbone needed for your run. The revisions below match the
experiment checkpoints.

```bash
hf download Qwen/Qwen3-VL-8B-Instruct \
  --revision 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
  --local-dir checkpoints/Qwen3-VL-8B-Instruct

# Optional: use this backbone for 32B evaluation.
hf download Qwen/Qwen3-VL-32B-Instruct \
  --revision b282fe1ca05915cc8825369f6aa510d7a0982594 \
  --local-dir checkpoints/Qwen3-VL-32B-Instruct
```

Model cards: [8B](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct),
[32B](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct).
Preparing question-dependent video frame selections additionally uses
[CLIP ViT-B/32](https://huggingface.co/openai/clip-vit-base-patch32);
it is unnecessary when evaluating already prepared observer reports.

## Datasets

Download the annotations, videos, and available official subtitle files using
the corresponding dataset instructions; downloading annotations alone does not
provide the video assets.

| Dataset         | Instructions                                                 | Download                                                     |
| --------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| LongVideoBench  | [Official repository](https://github.com/longvideobench/LongVideoBench) | [Hugging Face](https://huggingface.co/datasets/longvideobench/LongVideoBench) |
| Video-MME       | [Official repository](https://github.com/MME-Benchmarks/Video-MME) | [Hugging Face](https://huggingface.co/datasets/lmms-eval/Video-MME) |
| EgoSchema       | [Official repository](https://github.com/egoschema/EgoSchema) | [Official download instructions](https://github.com/egoschema/EgoSchema#-downloading-the-dataset) |
| MLVU (optional) | [Official repository](https://github.com/JUNJIE99/MLVU)      | [Dev](https://huggingface.co/datasets/MLVU/MVLU) / [Test](https://huggingface.co/datasets/MLVU/MLVU_Test) |
