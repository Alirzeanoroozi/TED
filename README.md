# TED: Transformer-based Protein Domain Classification

A comprehensive machine learning project for protein domain classification using transformer architectures. TED (Transformer-based Domain classification) leverages state-of-the-art language models to predict protein domain boundaries and CATH classifications from protein sequences.

## 🧬 Overview

This project implements multiple approaches for protein domain classification:

- **From-scratch Transformer**: Custom transformer implementation for sequence-to-sequence domain prediction
- **LoRA Fine-tuning**: Efficient fine-tuning of pre-trained language models using Low-Rank Adaptation
- **ESM Integration**: Protein-specific embeddings using Evolutionary Scale Modeling (ESM)

The system takes protein sequences as input and predicts:
- Domain boundaries (start/end positions)
- CATH structural classifications
- Multi-domain protein parsing

## ��️ Project Structure

```
TED/
├── transformer_scratch/          # Custom transformer implementation
│   ├── model.py                 # Transformer architecture
│   ├── train.py                 # Training script
│   ├── dataloader.py            # Data loading utilities
│   └── tokenizers/              # Custom tokenizers
├── LoRA/                        # LoRA fine-tuning approach
│   ├── train.py                 # LoRA training script
│   ├── inference.py             # Inference pipeline
│   └── extended_tokenized_ds/   # Preprocessed datasets
├── esm/                         # ESM embeddings and analysis
│   ├── create_embeddings.py     # ESM embedding generation
│   └── embeddings/              # Pre-computed embeddings
├── dlp/                         # Data loading and preprocessing
│   ├── create_data.py           # Dataset creation
│   └── jsons/                   # Training/validation data
├── configs/                     # Configuration files
├── results/                     # Analysis and evaluation
└── data/                        # Raw data and exports
```

## 🚀 Quick Start

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd TED
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up data (see Data Setup section below)

### Training

#### From-scratch Transformer
```bash
cd transformer_scratch
python train.py
```

#### LoRA Fine-tuning
```bash
cd LoRA
python train.py --config ../configs/transformer_config.yaml
```

### Inference

#### Using LoRA model
```bash
cd LoRA
python inference.py --model_path <path_to_model> --sequence "MCCLTSILPLAALAADAEK..."
```

## 📊 Data Format

The system expects protein sequences with domain annotations in the following format:

**Input**: Protein sequence (amino acid string)
```
MCCLTSILPLAALAADAEKAPATTEAPAAEAPRPPLLERSQEDALALERLVPRAEQQTLQAGADSFLALWKPANDSDPQGAVIIVPGAGETADWPNAVGPLRQKFPDVGWHSLSLSLPDLLADSPQARVEAKPAAEPEKTKGESAPAKDVPADANANVAQATAADADTAESTDAEQASEQTDTADAERIFARLDAAVAFAQQHNARSIVLIGHGSGAYWAARYLSEKQPPHVQKLVMVAAQTPARVEHDLESLAPTLKVPTADIYYATRSQDRSAAQQRLQASKRQKDSQYRQLSLIAMPGNKAAEQEQLFRRVRGWMSPQG
```

**Output**: Domain boundaries and CATH classifications
```
1-50_100-150 | 1.10.8.10 | 200-250 | 2.40.50.140
```

Where:
- `1-50_100-150`: Domain boundaries (start-end positions)
- `1.10.8.10`: CATH classification (Class.Architecture.Topology.Homology)

## 📈 Configuration

### Transformer Configuration
Key parameters in `configs/transformer_config.yaml`:

```yaml
# Model Architecture
num_encoder_layers: 6
num_decoder_layers: 6
emb_size: 512
nhead: 8

# Training
learning_rate: 1e-4
per_device_train_batch_size: 8
num_train_epochs: 1

# Data
src_max_seq_len: 2048
tgt_max_seq_len: 128
```

### LoRA Configuration
```yaml
# LoRA Parameters
r: 16
lora_alpha: 32
lora_dropout: 0.1
target_modules: ["q", "v", "k", "o", "wi_0", "wi_1", "wo"]

# Quantization
load_in_4bit: true
bnb_4bit_quant_type: "nf4"
```

## 📈 Results and Evaluation

The project includes comprehensive evaluation metrics:

- **Domain Boundary Accuracy**: Precision of domain start/end predictions
- **CATH Classification Accuracy**: Correctness of structural classifications
- **Multi-domain Parsing**: Handling of proteins with multiple domains

Results are stored in `results/` directory with analysis scripts for detailed evaluation.

## 🧪 ESM Integration

The project leverages ESM (Evolutionary Scale Modeling) for protein-specific embeddings:

```python
from esm.sdk.api import ESMProtein, LogitsConfig

# Generate protein embeddings
protein = ESMProtein(sequence=sequence)
protein_tensor = client.encode(protein)
logits_output = client.logits(protein_tensor, LogitsConfig(sequence=True, return_embeddings=True))
```

## 📚 Dependencies

- `torch` - PyTorch for deep learning
- `transformers` - Hugging Face transformers library
- `peft` - Parameter-Efficient Fine-Tuning
- `datasets` - Dataset handling
- `esm` - Evolutionary Scale Modeling
- `wandb` - Experiment tracking
- `bitsandbytes` - Quantization support

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Contact

For questions or contributions, please open an issue or contact the maintainers.

---

**Note**: This project is for research purposes. Please ensure you have appropriate data usage rights for any protein datasets you use.
```

This README provides a comprehensive overview of your TED project, covering:

1. **Clear project description** - What TED does and its purpose
2. **Architecture overview** - The different approaches implemented
3. **Project structure** - Well-organized directory layout
4. **Quick start guide** - Installation and basic usage
5. **Data format examples** - Clear input/output specifications
6. **Configuration details** - Key parameters and settings
7. **Results and evaluation** - How to assess model performance
8. **Technical features** - Key capabilities and integrations
9. **Dependencies** - Required packages
10. **Contributing guidelines** - How others can contribute

The README is structured to be both informative for researchers and practical for users who want to run the code. It highlights the unique aspects of your project, particularly the multi-approach implementation and protein-specific adaptations.
