# Finetune-Mistral-7B-Indian-CS-Compliance-LoRA-4bit-Quantized

A specialized legal AI assistant fine-tuned on Indian corporate law, optimized for Company Secretary (CS) compliance perspectives. This project adapts the Mistral-7B-v0.1 model using parameter-efficient fine-tuning techniques to provide accurate, compliance-focused legal guidance under Indian law.

## 🎯 Project Overview

This project demonstrates advanced LLM fine-tuning techniques applied to the legal domain, specifically targeting Indian Company Secretary compliance requirements. By leveraging QLoRA and 4-bit quantization, the model achieves domain specialization while maintaining computational efficiency.

## ✨ Key Features

- **Base Model**: Mistral-7B-v0.1 (7 billion parameters)
- **Dataset**: [169Pi Indian Law Dataset](https://huggingface.co/datasets/169Pi/indian_law) (47.8k examples)
- **Fine-tuning Method**: QLoRA (Quantized Low-Rank Adaptation)
- **Quantization**: 4-bit for memory efficiency
- **Domain Focus**: Company Secretary compliance under Indian corporate law
- **Training Hardware**: Lightning AI H200 GPU
- **Training Time**: ~3-4 hours

## 🛠️ Technical Implementation

### Model Architecture
- **LoRA Configuration**:
  - Rank (r): 8
  - Alpha: 16
  - Target modules: Query and Value projection layers
- **Quantization**: 4-bit quantization using bitsandbytes
- **Framework**: Hugging Face Transformers

### Training Configuration

```python
# LoRA Parameters
lora_r = 8
lora_alpha = 16
lora_dropout = 0.05

# Quantization
load_in_4bit = True
bnb_4bit_compute_dtype = "float16"
bnb_4bit_quant_type = "nf4"
```

### Instruction Format

The model is trained with a custom instruction format to bias responses toward CS compliance perspectives:

```python
def to_instruction_format(example):
    return {
        "instruction": (
            "Explain the following from a Company Secretary compliance "
            "perspective under Indian law"
        ),
        "input": example["prompt"],
        "output": example["response"]
    }
```

## 📊 Dataset

- **Source**: [169Pi/indian_law](https://huggingface.co/datasets/169Pi/indian_law)
- **Size**: 47,800 rows
- **Content**: Indian legal text covering:
  - Corporate governance
  - Regulatory compliance
  - Statutory requirements
  - Company Secretary responsibilities
  - Indian corporate law provisions

## 🚀 Getting Started

### Prerequisites

```bash
pip install torch transformers accelerate bitsandbytes peft datasets
```

### Installation

```bash
git clone https://github.com/yourusername/Finetune-Mistral-7B-Indian-CS-Compliance-LoRA-4bit-Quantized.git
cd Finetune-Mistral-7B-Indian-CS-Compliance-LoRA-4bit-Quantized
pip install -r requirements.txt
```

## 💡 Use Cases

- **Compliance Queries**: Company Secretary regulatory requirements
- **Corporate Governance**: Board meeting procedures, resolutions, and governance best practices
- **Statutory Compliance**: ROC filings, annual returns, and statutory registers
- **Legal Research**: Quick reference for Indian corporate law provisions
- **Training**: Educational tool for CS professionals and law students

## 🔧 Technical Advantages

### Parameter-Efficient Fine-Tuning
- **Memory Efficient**: 4-bit quantization reduces memory footprint by ~75%
- **Fast Training**: LoRA adapters train only ~0.5% of total parameters
- **Cost Effective**: Enables fine-tuning on consumer-grade hardware

### Performance Optimizations
- Quantized inference for faster deployment
- Low-rank adaptation preserves base model knowledge
- Efficient adapter merging for production use

## 🎓 Learning Outcomes

This project demonstrates:
- Large Language Model fine-tuning techniques
- Parameter-efficient training methods (LoRA/QLoRA)
- Model quantization for resource optimization
- Domain-specific NLP application
- Legal tech / RegTech implementation
- Hugging Face ecosystem proficiency

## ⚖️ Legal Disclaimer

This model is designed for educational and informational purposes. It should not be considered as legal advice or a substitute for professional legal consultation. Always consult with qualified legal professionals for specific compliance requirements.

## 🔮 Future Enhancements

- [ ] Model evaluation metrics and benchmarks
- [ ] Deployment on Hugging Face Hub
- [ ] Web interface for interactive queries
- [ ] Multi-lingual support (Hindi, other regional languages)
- [ ] Integration with legal document databases
- [ ] RAG (Retrieval-Augmented Generation) implementation

## 📝 Citation

If you use this work, please cite:

```bibtex
@software{indian_cs_compliance_llm,
  title={Fine-tuned Mistral-7B for Indian Company Secretary Compliance},
  author={Swapnil Khot},
  year={2025},
  url={https://github.com/mobndash/Finetune-Mistral-7B-v0.1-Indian-Company-Secretary-Law-LoRA-QLoRA-4bit}
}
```
## 🙏 Acknowledgments

- [Mistral AI](https://mistral.ai/) for the base model
- [169Pi](https://huggingface.co/169Pi) for the Indian Law dataset
- [Hugging Face](https://huggingface.co/) for the transformers library
- [Lightning AI](https://lightning.ai/) for H200 GPU access
