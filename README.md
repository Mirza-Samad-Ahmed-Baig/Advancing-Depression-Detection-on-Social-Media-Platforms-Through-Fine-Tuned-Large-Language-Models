# Advancing-Depression-Detection-on-Social-Media-Platforms-Through-Fine-Tuned-Large-Language-Models

# Llama-2-7b Fine-tuning

This repository contains the code for fine-tuning a Llama-2 model for depression detection. The approach is used in the paper **"Advancing Depression Detection on Social Media Platforms Through Fine-Tuned Large Language Models"**.

The model uses Hugging Face Transformers and PEFT (Parameter Efficient Fine-Tuning) to efficiently train large models on custom datasets, optimizing for performance and memory efficiency.

## Table of Contents

- [Requirements](#requirements)
- [Setup](#setup)
- [Dataset](#dataset)
- [Model Training](#model-training)
- [Running the Model](#running-the-model)
- [TensorBoard](#tensorboard)

## Requirements

Before running this code, ensure that you have the following installed:

- Python 3.7+
- Jupyter Notebook or Google Colab
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [PEFT](https://github.com/huggingface/peft)
- [BitsAndBytes](https://github.com/facebookresearch/bitsandbytes)
- [Accelerate](https://huggingface.co/docs/accelerate)
- [TensorBoard](https://www.tensorflow.org/tensorboard)
- [Google Colab](https://colab.research.google.com/) (Optional but recommended for GPU support)

### Installation

You can install the required libraries using the following command:

```bash
pip install accelerate peft bitsandbytes transformers trl
```

## Setup

### 1. Clone the repository:

```bash
git clone repository-name
cd repository-name
```

### 2. Mount Google Drive (Optional)

If you're running this on Google Colab, you can mount your Google Drive to store the fine-tuned models and datasets:

```python
from google.colab import drive
drive.mount('/content/drive')
```

### 3. Set up the paths

Replace any placeholder paths like `/yourpath/foldername/` with the actual paths on your system or Google Drive.

## Dataset

The model is fine-tuned using a custom dataset available on the Hugging Face Hub. You can replace this with any dataset of your choice.

For example:
```python
dataset_name = "yourpath/dataset_name"
```

Ensure that the dataset is in the required format for training and evaluation.

## Model Training

The training process fine-tunes a Llama-2 model using the LoRA approach to reduce computational overhead and memory usage while maintaining high performance. This method is part of the work described in **"Advancing Depression Detection on Social Media Platforms Through Fine-Tuned Large Language Models"**.

Run the following code to start the fine-tuning process:

```python
trainer.train()
```

The model will be fine-tuned for 20 epochs by default. You can change the number of epochs or other hyperparameters in the `TrainingArguments` section of the code.

### Saving the Fine-Tuned Model

Once the training is complete, the model will be saved to the specified path:

```python
trainer.model.save_pretrained("/yourpath/foldername/pad_model")
trainer.tokenizer.save_pretrained("/yourpath/foldername/pad_model")
```

## Running the Model

After fine-tuning, you can run the trained model for text generation. Here’s how to use it for generating text:

```python
prompt = "Well well well, see who slept again for 14 hours :( ?"
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=2000)
result = pipe(f"<s>[INST] {prompt} [/INST]")
print(result[0]['generated_text'])
```

## TensorBoard

To monitor the training process, you can use TensorBoard. Run the following code to start TensorBoard in Google Colab:

```python
from tensorboard import notebook
log_dir = "/yourpath/foldername/output_dir/runs"
notebook.start("--logdir {} --port 4000".format(log_dir))
```

