# 🧠 Reducing Inference Time in GPT Models

This project benchmarks attention mechanism optimizations—**FlashAttention**, **ALiBi (Attention with Linear Biases)**, and **Sparse Attention**—within the GPT-2 architecture to reduce inference time while preserving summarization quality.

Built using **PyTorch** and **HuggingFace Transformers**, the model is evaluated on the **News Dataset** for summarization, with comparisons across latency, ROUGE/BLEU scores, and GPU memory usage.

---

## 🚀 Key Features

- 🔁 **FlashAttention 2.4**: GPU-optimized attention kernel for speed and efficiency  
- 🧭 **ALiBi**: Adds linear bias for positional awareness and long-context generalization  
- 📉 **Sparse Attention**: Sliding window pattern to reduce attention computation  
- 🧪 **Benchmarked** on Colab Pro A100 with competetive GPU usage  
- 📊 **Evaluation Metrics**: ROUGE-1/2/L, BLEU, inference latency, GPU memory

---

## 📂 Repository Structure
├── model/
│ ├── gpt2_flash.py # FlashAttention integration
│ ├── gpt2_alibi.py # ALiBi positional bias logic
│ └── gpt2_sparse.py # Sparse attention window mask
├── data/
│ └── news_dataset.py # Preprocessing logic for News Summary dataset
├── evaluate/
│ └── metrics.py # ROUGE, BLEU, and timing functions
├── hpml_final.ipynb # Full training + evaluation notebook
├── plots/ # Performance visualizations
└── README.md


---

## 📈 Results Summary

| Variant           | ROUGE-L | BLEU   | Inference Time (ms) | Memory (GB) |
|------------------|---------|--------|----------------------|-------------|
| Standard GPT-2    | 0.1215  | 0.0263 | 96.22                | ~20.9       |
| Flash GPT-2       | 0.1076  | 0.0221 | 73.42                | ~20.9       |
| Flash + ALiBi     | 0.1320  | 0.0295 | 87.08                | ~20.9       |
| Flash + Sparse    | 0.1303  | 0.0282 | 86.86                | ~20.9       |

---

bash--
jupyter notebook hpml_final.ipynb

🙌 Acknowledgements
FlashAttention by Tri Dao et al.

ALiBi by Press et al.

HuggingFace Transformers and Datasets





