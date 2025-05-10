👥 Authors
Nikhil Bhise – nb4053@nyu.edu

Rudra Patil – rp4216@nyu.edu

Developed as part of the final project for ECE-GY 9143: High Performance Machine Learning, Spring 2025 – New York University.




# 🚀 Reducing Inference Time in GPT Models

This project explores performance optimizations for Transformer-based summarization using GPT-2. We integrate and evaluate **FlashAttention**, **ALiBi (Attention with Linear Biases)**, and **Sparse Attention** to improve inference time, memory efficiency, and scalability—without compromising output quality.

---

## 📌 Project Overview

Transformer models like GPT-2 offer excellent summarization quality but suffer from high memory consumption and slow inference—especially under deployment constraints. In this project, we re-engineered GPT-2's attention mechanism to test three modern techniques:

- 🔥 **FlashAttention** – A CUDA kernel-fused implementation that reduces memory access and accelerates attention computation.
- 🧭 **ALiBi** – A lightweight positional encoding mechanism using linear biases instead of embeddings, improving generalization to longer inputs.
- 🧠 **Sparse Attention** – Reduces computational complexity by restricting attention to a local context window.



## 🎯 Project Milestones

| Milestone                          | Status       |
|-----------------------------------|--------------|
| Setup GPT-2 with HuggingFace      | ✅ Completed |
| Integrated FlashAttention kernel  | ✅ Completed |
| Implemented ALiBi positional bias | ✅ Completed |
| Added SparseAttention variant     | ✅ Completed |
| Training on summarization dataset | ✅ Completed |
| Evaluation & benchmarking         | ✅ Completed |
| Final presentation + documentation| ✅ Completed |

---

## 📂 Repository Structure

├── model/
│   ├── flash_attention.py       # FlashAttention variant
│   ├── alibi_attention.py       # ALiBi implementation
│   └── sparse_attention.py      # Custom SparseAttention block
├── data/
│   └── preprocess.py            # Dataset loading and tokenization
├── evaluation/
│   └── metrics.py               # ROUGE/BLEU scoring, latency tracking
├── plots/                       # Inference and quality visualizations
├── hpml_final.ipynb             # Full training and analysis notebook
└── README.md


Run this project using Colab Pro (with A100 GPU):

# Setup environment (Colab)
!pip install flash-attn==2.4.2 --no-build-isolation
!pip install transformers datasets evaluate rouge-score

# Run notebook
!jupyter notebook hpml_final.ipynb


----📊 Results Summary ----

🧠 Summarization Quality (ROUGE/BLEU) : 

| Model          | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU   |
| -------------- | ------- | ------- | ------- | ------ |
| Standard GPT-2 | 0.1806  | 0.0804  | 0.1215  | 0.0263 |
| Flash GPT-2    | 0.1608  | 0.0691  | 0.1076  | 0.0221 |
| Flash + ALiBi  | 0.1975  | 0.0868  | 0.1320  | 0.0295 |
| Flash + Sparse | 0.1943  | 0.0856  | 0.1303  | 0.0282 |


⚡ Inference Time (ms) : 

| Model          | Inference Time |
| -------------- | -------------- |
| Standard GPT-2 | 96.22 ms       |
| Flash GPT-2    | 73.42 ms       |
| Flash + ALiBi  | 87.08 ms       |
| Flash + Sparse | 86.86 ms       |


💾 Memory Usage (MB) : 

| Model          | Memory Usage |
| -------------- | ------------ |
| Standard GPT-2 | 9465.81 MB   |
| Flash GPT-2    | 8867.81 MB   |
| Flash + ALiBi  | 8865.81 MB   |
| Flash + Sparse | 8877.81 MB   |


⏱️ Training Time (Batch Size Impact) :

| Batch Size | Standard (ms) | Flash (ms) | Speedup (%) |
| ---------- | ------------- | ---------- | ----------- |
| 16         | 1508.43       | 1467.35    | 2.62%       |
| 32         | 1353.53       | 1334.77    | 1.39%       |
| 64         | 1325.98       | 1283.22    | 3.22%       |
| 128        | 1341.39       | 1274.07    | 5.07%       |



----🔍 Observations ----
⚡ FlashAttention significantly reduced inference time (~24%) and memory usage (~6%).

🧠 Flash + ALiBi achieved the highest summarization quality but added some latency.

⚖️ Flash + Sparse offered the best trade-off between quality and speed.

🔁 FlashAttention’s benefits increased with larger batch sizes, making it ideal for high-throughput training.



----📎 Citation ----
If you find this useful, please cite our project or use the code with attribution.
Developed for ECE-GY 9143: High Performance Machine Learning, Spring 2025 @ NYU.


----🔗 Project Repository ----
🔗 GitHub: https://github.com/Nikb033/Reducing-Inference-Time-in-GPT-Models



