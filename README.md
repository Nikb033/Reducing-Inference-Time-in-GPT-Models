# 🧠 Reducing Inference Time in GPT Models

This project benchmarks attention mechanism optimizations—**FlashAttention**, **ALiBi (Attention with Linear Biases)**, and **Sparse Attention**—within the GPT-2 architecture to reduce inference time while preserving summarization quality.

Built using **PyTorch** and **HuggingFace Transformers**, the model is evaluated on the **News Dataset** for summarization, with comparisons across latency, ROUGE/BLEU scores, and GPU memory usage.

---

## 🚀 Key Features

- 🔁 **FlashAttention 2.4**: GPU-optimized attention kernel for speed and efficiency  
- 🧭 **ALiBi**: Adds linear bias for positional awareness and long-context generalization  
- 📉 **Sparse Attention**: Sliding window pattern to reduce attention computation  
- 🧪 **Benchmarked** on Colab Pro A100 with consistent 20.9GB GPU usage  
- 📊 **Evaluation Metrics**: ROUGE-1/2/L, BLEU, inference latency, GPU memory

---

## 📂 Repository Structure

