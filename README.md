# 🏥 Llama3 Medical Finetune

> Fine-tune **Llama 3.2** on medical Q&A data using **QLoRA + Unsloth** — runs entirely on a **free Google Colab T4 GPU**.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)
![Colab](https://img.shields.io/badge/Google%20Colab-T4%20GPU-orange?style=flat-square&logo=googlecolab)
![Unsloth](https://img.shields.io/badge/Unsloth-QLoRA-green?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-brightgreen?style=flat-square)

---

## 🧠 What This Project Does

Takes **Llama 3.2 (3B)** — a general-purpose AI — and fine-tunes it into a **medical specialist** that accurately answers clinical questions like:

- *"What are the symptoms of Type 2 Diabetes?"*
- *"What is the mechanism of action of Aspirin?"*
- *"How is pneumonia diagnosed?"*

All of this runs on a **free** Google Colab GPU using memory-efficient QLoRA — no expensive hardware needed.

---

## ⚡ Key Techniques

| Technique | What It Does |
|-----------|-------------|
| **QLoRA** | Trains only ~0.7% of model parameters instead of all 3B |
| **4-bit Quantization** | Shrinks model from 12GB → 3GB in GPU memory |
| **Unsloth** | Makes training 2× faster with 60% less memory |
| **LoRA Adapters** | Small plugin layers that learn medical knowledge |
| **SFTTrainer** | Handles the full training loop automatically |

---

## 🗂️ Project Structure

```
llama3-medical-finetune/
├── Medical_Finetuning_QLoRA_Unsloth.ipynb   # Main Colab notebook
├── README.md                                 # You're here
└── medical_lora_adapter/                     # Saved after training
    ├── adapter_model.safetensors             # Trained weights
    ├── adapter_config.json                   # LoRA configuration
    └── tokenizer files...
```

---

## 🚀 How to Run

### 1. Open in Google Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

- Go to [colab.research.google.com](https://colab.research.google.com)
- Click **File → Upload Notebook**
- Upload `Medical_Finetuning_QLoRA_Unsloth.ipynb`

### 2. Enable GPU
```
Runtime → Change runtime type → T4 GPU → Save
```

### 3. Run All Cells in Order

| Cell | What It Does | Time |
|------|-------------|------|
| 1 | Check GPU | ~5s |
| 2 | Install Unsloth & libraries | ~3 min |
| 3 | Import everything | ~20s |
| 4 | Load Llama 3.2 (4-bit) | ~5 min |
| 5 | Attach LoRA adapters | ~10s |
| 6 | Load medical dataset | ~30s |
| 7 | Format prompts | ~20s |
| 8 | Configure trainer | ~5s |
| 9 | **Train the model** ⬅️ | ~15-20 min |
| 10 | Save adapter | ~20s |
| 11 | Test with medical questions | ~1 min |

---

## 📊 Training Details

```
Base Model    : unsloth/Llama-3.2-3B-Instruct
Dataset       : medalpaca/medical_meadow_medical_flashcards
Samples used  : 2,000 medical Q&A pairs
Epochs        : 3
LoRA Rank     : 16
Learning Rate : 2e-4
Batch Size    : 2 (effective: 8 with gradient accumulation)
GPU Memory    : ~6 GB / 15 GB (T4)
Training Time : ~15-20 minutes
Adapter Size  : ~100 MB
```

---

## 💊 Sample Output

**Question:** What are the symptoms of Type 2 Diabetes?

**Fine-tuned Model Answer:**
> Type 2 Diabetes Mellitus presents with classic symptoms including polyuria (frequent urination), polydipsia (excessive thirst), polyphagia (increased hunger), fatigue, blurred vision, and slow wound healing. Patients may also experience recurrent infections and numbness or tingling in the extremities due to peripheral neuropathy...

---

## 🔁 Loading Your Saved Adapter Later

```python
from unsloth import FastLanguageModel
from peft import PeftModel

# Load base model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name   = "unsloth/Llama-3.2-3B-Instruct",
    max_seq_length = 1024,
    load_in_4bit   = True,
)

# Attach your saved adapter
model = PeftModel.from_pretrained(model, "./medical_lora_adapter")
FastLanguageModel.for_inference(model)
print("✅ Medical model reloaded!")
```

---

## 📦 Requirements

No local installation needed — everything runs in Google Colab. Libraries used:

```
unsloth
trl
peft
datasets
transformers
accelerate
bitsandbytes
torch
```

---

## ⚠️ Disclaimer

> This model is for **educational purposes only**. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider for medical decisions.

---

## 🤝 Contributing

Pull requests welcome! Ideas for improvement:
- Train on larger dataset (remove the 2000 sample cap)
- Try different base models (Mistral, DeepSeek-R1, Gemma3)
- Add evaluation on MedQA benchmark
- Build a Streamlit chat interface on top

---

## 📄 License

MIT © [Hamama Komal](https://github.com/Hamama-Komal)

---

<div align="center">
  Built with ❤️ using <a href="https://github.com/unslothai/unsloth">Unsloth</a> + <a href="https://colab.research.google.com">Google Colab</a>
  <br><br>
  <b>⭐ Star this repo if it helped you!</b>
</div>
