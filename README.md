# GPT-OSS-20B Multilingual Fine-tuning with DeepSpeed

This project fine-tunes OpenAI's GPT-OSS-20B model on multilingual reasoning tasks using LoRA (Low-Rank Adaptation), TRL (Transformer Reinforcement Learning), and DeepSpeed for distributed training.

## 🚀 Features

- **DeepSpeed Integration**: Distributed training across multiple GPUs
- **LoRA Fine-tuning**: Memory-efficient parameter adaptation
- **Multilingual Support**: Training on multilingual reasoning datasets
- **HuggingFace Hub Integration**: Optional model uploading
- **UV Package Management**: Fast dependency management
- **TensorBoard Logging**: Real-time training metrics

## 📁 Project Structure

```
proj/
├── README.md                   # This file
├── pyproject.toml              # UV project configuration (will be created after `uv init`)
├── uv.lock                     # UV lockfile (will be created after `uv init`)
├── main.py                     # 🎯 Main launcher script
├── train_ds.py                 # 🔥 DeepSpeed training script
├── ds_config.json              # ⚙️ DeepSpeed configuration
└── tensorboard_logs/           # 📊 Training logs (created during training)
```

## 📄 Script Explanations

### `main.py` - Launcher Script 🎯
The main entry point that handles:
- **Command-line argument parsing** (GPU configuration, HF tokens)
- **Environment variable management** (HF authentication, feature flags)
- **DeepSpeed command execution** (automatically runs `deepspeed --num_gpus=4 train_ds.py`)
- **Requirement checking** (verifies DeepSpeed installation and required files)
- **Real-time output streaming** (shows training progress)

**Key Features:**
- Automatic DeepSpeed command generation
- HF token management via CLI or environment variables
- Flexible GPU configuration (specific GPUs or count)
- Training process monitoring with error handling

### `train_ds.py` - Training Script 🔥
The core training implementation featuring:
- **Model Loading**: OpenAI GPT-OSS-20B with BFloat16 precision
- **LoRA Configuration**: Low-rank adaptation for specific model layers
- **Dataset Processing**: HuggingFace multilingual thinking dataset
- **DeepSpeed Integration**: ZeRO Stage 2 optimization
- **HF Hub Integration**: Optional model pushing with smart authentication
- **Distributed Training**: Multi-GPU coordination and rank management

**Key Components:**
- `load_data()`: Loads multilingual reasoning dataset
- `apply_lora()`: Configures LoRA adapters for efficient fine-tuning
- `train()`: Main training loop with DeepSpeed optimization
- `evaluate()`: Post-training evaluation on test prompts
- `setup_huggingface_auth()`: Smart HF authentication handling

### `ds_config.json` - DeepSpeed Configuration ⚙️
DeepSpeed optimization settings:
- **ZeRO Stage 2**: Gradient and optimizer state partitioning
- **BFloat16**: Mixed precision training
- **Memory Optimization**: Gradient accumulation and checkpointing
- **Communication**: Overlapped communication for efficiency
- **TensorBoard**: Integrated logging configuration

## 🛠️ Installation & Setup

### Prerequisites
- **Hardware**: 4x NVIDIA H200 SXM GPUs (or similar high-end GPUs)
- **Platform**: RunPod cloud instance (recommended)
- **Base Image**: `runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04`
- **System**: 96 vCPU, 1511 GB RAM
- **Python**: 3.11+ (included in RunPod image)
- **UV**: Package manager for dependency management

### 1. Install UV (if not already installed)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Clone and Setup Project
```bash
git clone <your-repo-url>
cd proj
uv sync  # Install dependencies from lockfile
```

### 3. RunPod Environment Setup
The RunPod PyTorch image comes pre-configured with:
- ✅ CUDA 12.8.1 and cuDNN
- ✅ PyTorch 2.8.0 
- ✅ Python 3.11
- ✅ Ubuntu 22.04

No additional CUDA setup required!

### 4. Get Hugging Face Token (Optional)
1. Visit: https://huggingface.co/settings/tokens
2. Create a new token with "Write" permissions
3. Copy the token for later use

## 🚀 Usage

### Basic Training (4 GPUs)
```bash
uv run main.py --hf_token your_hf_token_here
```

### Advanced Usage Examples

#### Custom GPU Configuration
```bash
# Use 2 GPUs
uv run main.py --num_gpus=2 --hf_token your_token

# Use specific GPUs
uv run main.py --include_gpus=0,1,2,3 --hf_token your_token
```

#### Training Options
```bash
# Disable HuggingFace Hub upload
uv run main.py --no_push_to_hub

# Skip post-training evaluation (faster)
uv run main.py --no_evaluation

# Train without HF token (local only)
uv run main.py --no_push_to_hub
```

#### Environment Variable Approach
```bash
# Set token via environment variable
export HF_TOKEN=your_hf_token_here
uv run main.py

# Disable features via environment
export PUSH_TO_HUB=false
export RUN_EVALUATION=false
uv run main.py
```

## 📊 Monitoring Training

### TensorBoard
```bash
# In a separate terminal
tensorboard --logdir=./tensorboard_logs
```
Then visit: http://localhost:6006

### Real-time Logs
Training progress is displayed in real-time in your terminal, including:
- Loss metrics
- Learning rate schedules  
- Memory usage
- Training speed (tokens/sec)

## ⚙️ Configuration

### Training Parameters (in `train_ds.py`)
```python
learning_rate = 2e-4
num_train_epochs = 1
per_device_train_batch_size = 2
gradient_accumulation_steps = 8
max_length = 2048
```

### LoRA Parameters
```python
r = 8                    # Rank
lora_alpha = 16         # Alpha scaling
target_parameters = [   # Specific layers to adapt
    "7.mlp.experts.gate_up_proj",
    "7.mlp.experts.down_proj", 
    # ... more layers
]
```

### DeepSpeed Configuration
- **ZeRO Stage 2**: Optimal for 4x H200 SXM GPUs
- **Effective Batch Size**: 64 (2 × 4 GPUs × 8 accumulation)
- **Memory**: BFloat16 precision for efficiency
- **H200 Optimized**: Configured for high-bandwidth HBM3e memory
- **RunPod Ready**: Compatible with pre-installed CUDA 12.8 environment

## 🔧 Troubleshooting

### Common Issues

#### DeepSpeed Installation
```bash
# If DeepSpeed is missing
uv add deepspeed
```

#### CUDA/GPU Issues
```bash
# Check GPU availability
nvidia-smi

# Verify CUDA installation
python -c "import torch; print(torch.cuda.is_available())"
```

#### Memory Issues
- Reduce `per_device_train_batch_size` from 2 to 1
- Increase `gradient_accumulation_steps` to maintain effective batch size
- Enable ZeRO Stage 3 in `ds_config.json` for larger models

#### HF Authentication Errors
- Verify token has "Write" permissions
- Check token is not expired
- Use `--no_push_to_hub` to train locally without HF Hub

## 🎯 Expected Results

### Training Output
- **Duration**: ~30-60 minutes on 4x H200 SXM (significantly faster than consumer GPUs)
- **Memory Usage**: ~20-40GB per GPU (out of 141GB available per H200)
- **Final Model**: LoRA adapters + base model
- **Artifacts**: Model checkpoints + TensorBoard logs

### Model Performance
- Fine-tuned for multilingual reasoning tasks
- Improved performance on German, Spanish, and other languages
- Maintains base model capabilities while adding specialized reasoning

### Hardware Benefits (H200 SXM)
- **High Memory Bandwidth**: 4.8TB/s per GPU for faster training
- **Large VRAM**: 141GB HBM3e per GPU allows larger batch sizes
- **NVLink**: Direct GPU-to-GPU communication for optimal multi-GPU scaling
- **RunPod Optimization**: Pre-configured environment with CUDA 12.8 and cuDNN

## 📝 Notes

- **Dataset**: Uses HuggingFaceH4/Multilingual-Thinking
- **Base Model**: OpenAI GPT-OSS-20B (20 billion parameters)
- **Method**: LoRA fine-tuning (trains only ~0.1% of parameters)
- **Output Format**: PEFT adapter that can be merged with base model

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch
3. Make your changes
4. Test with `uv run main.py --no_push_to_hub`
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Happy Training! 🚀**
