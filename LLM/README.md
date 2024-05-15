
# LLM

This repository contains scripts and resources for fine-tuning the Llama2 model with LoRA, evaluating its performance, and processing the output for punctuation restoration. The primary components include environment setup, model fine-tuning, output processing, and performance evaluation.

## Table of Contents
- [Installation](#installation)
- [Getting Started](#getting-started)
- [File Descriptions](#file-descriptions)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Installation

### Setting Up the Conda Environment
To set up the Conda environment for this project, you will need to have Conda installed on your system. Once Conda is installed, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/LLM.git
   cd LLM
   ```

2. Create the Conda environment using the provided `environment.yml` file:
   ```bash
   conda env create -f environment.yml
   ```

3. Activate the Conda environment:
   ```bash
   conda activate llm_env
   ```

### Acquiring Access to Llama2 Model
To use the Llama2 model, you need to acquire access to the model weights from Hugging Face. Follow the instructions on the [Hugging Face website](https://huggingface.co/) to request access and obtain the necessary credentials.

## Getting Started

After setting up the Conda environment and acquiring the necessary model weights, you can start using the scripts provided in this repository.

## File Descriptions

### `fine_tune_llama.py`
This script is used for fine-tuning the Llama2 model with LoRA. It also includes functionalities for testing the fine-tuned model.

### `extract_n_reinsert.py`
This script processes the output from the fine-tuned Llama2 model and Gemini by extracting and reinserting punctuation into the ground truth (with punctuation removed).

### `calculate_WER.py`
This script measures the Word Error Rate (WER) and Sentence Error Rate (SER) to evaluate the modification rate of the LLM output compared to the input.

### `manual_eval_metrics.py`
This script evaluates the performance of punctuation restoration by calculating precision, recall, and F1 scores.

### `gemini.py`
This script contains functionalities related to the Gemini model, which may be involved in processing or evaluating the LLM outputs.

### `environment.yml`
This file specifies the dependencies and environment configuration needed to run the scripts in this repository.

## Usage

### Fine-Tuning Llama2
To fine-tune the Llama2 model using the `fine_tune_llama.py` script, run:
```bash
python fine_tune_llama.py --config config.json
```
Make sure to update the `config.json` with the appropriate parameters for your fine-tuning process.

### Extracting and Reinserting Punctuation
To process the output and reinsert punctuation using the `extract_n_reinsert.py` script, run:
```bash
python extract_n_reinsert.py --input output.txt --ground_truth ground_truth.txt
```

### Calculating WER and SER
To calculate the Word Error Rate (WER) and Sentence Error Rate (SER) using the `calculate_WER.py` script, run:
```bash
python calculate_WER.py --hypothesis hypothesis.txt --reference reference.txt
```

### Evaluating Punctuation Restoration
To evaluate the punctuation restoration performance using the `manual_eval_metrics.py` script, run:
```bash
python manual_eval_metrics.py --predictions predictions.txt --ground_truth ground_truth.txt
```

## Contributing

Contributions are welcome! Please read the [CONTRIBUTING.md](CONTRIBUTING.md) file for guidelines on how to contribute to this project.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
