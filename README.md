## System Requirements

- Multi-GPU machine (RunPod, Lambda Labs, AWS, or local)
- CUDA-compatible GPUs
- Python 3.9+

## Installation

1. **Install dependencies:**

```bash
pip install -r requirements.txt
```

2. **Configure environment variables:**

Create a `.env` file in the repository root:

```bash
HF_TOKEN=your-hf-token
HF_USERNAME=your-hf-username
WANDB_API_KEY=your-wandb-key
WANDB_PROJECT=your-project-name
WANDB_DISABLED=false
```

**Getting your tokens:**

- **HF_TOKEN**: Create at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
- **WANDB_API_KEY**: Find at [wandb.ai/authorize](https://wandb.ai/authorize)

3. **Accept model license:**

For Llama models, accept the license at [meta-llama/Llama-3.2-8B](https://huggingface.co/meta-llama/Llama-3.2-8B)

---

## 1. Distributed Data Parallelism with Accelerate

Train Llama 3.2 8B with QLoRA across multiple GPUs using Hugging Face Accelerate.

## Training

From the repository root:

```bash
# 1 GPU (baseline)
accelerate launch --config_file code/configs/accelerate/config_1gpu.yaml code/train_ddp_accelerate.py

# 2 GPUs
accelerate launch --config_file code/configs/accelerate/config_2gpu.yaml code/train_ddp_accelerate.py

# 4 GPUs
accelerate launch --config_file code/configs/accelerate/config_4gpu.yaml code/train_ddp_accelerate.py
```

**Outputs saved to:**

- `data/outputs/accelerate_ddp/<model_name>-1-gpu/`
- `data/outputs/accelerate_ddp/<model_name>-2-gpus/`
- `data/outputs/accelerate_ddp/<model_name>-4-gpus/`

## Evaluation

After training, evaluate each model (update the model name in the command below). You set it in the `code/config.yaml` file. Use lowercase for the model name.

```bash
# Evaluate 1 GPU model
python code/evaluate_qlora.py --adapter_path data/outputs/accelerate_ddp/<model_name>-1-gpu/lora_adapters

# Evaluate 2 GPU model
python code/evaluate_qlora.py --adapter_path data/outputs/accelerate_ddp/<model_name>-2-gpus/lora_adapters

# Evaluate 4 GPU model
python code/evaluate_qlora.py --adapter_path data/outputs/accelerate_ddp/<model_name>-4-gpus/lora_adapters
```

**Results saved alongside adapters:**

- `eval_results.json` - ROUGE scores
- `predictions.jsonl` - Model predictions
