# 🧠 LLM Reasoning Analysis: Do LLMs Mimic Graph Algorithms?

## 🚀 Overview

This project analyzes whether Large Language Models (LLMs) implicitly approximate classical graph traversal algorithms like **BFS and DFS**.

It combines interpretability techniques and hybrid planning to understand how LLMs reason in structured environments.

## 🎯 Why This Matters

LLMs often appear to reason, but their internal logic is unclear.
This project helps answer:

* Do LLMs follow structured reasoning like algorithms?
* Where do they fail?
* Can hybrid systems improve performance?

## 🧠 Approach

* Scratchpad-based reasoning evaluation
* Representational Similarity Analysis (RSA)
* Attention pattern analysis
* Hybrid symbolic + LLM planner (BFS/A*)

## ⚙️ Tech Stack

* Python
* PyTorch
* Hugging Face Transformers
* NumPy / SciPy

## 📊 Results

* LLMs show **partial similarity to BFS-like reasoning**
* Performance decreases on complex graphs
* Hybrid planner improves consistency and accuracy

## 🏗️ Architecture

Graph → LLM Reasoning → Analysis (RSA + Attention) → Hybrid Planner → Evaluation

## 📁 Project Structure

* `evaluation_runner.py` → Runs experiments
* `graphs.py` → Graph environments
* `planner.py` → Hybrid planner
* `attention_analysis.py` → Attention analysis
* `rsa_analysis.py` → Representational similarity
* `scratchpad_runner.py` → Step-by-step reasoning

## ▶️ How to Run

```bash
pip install -r requirements.txt
python run_capstone_transformers.py
python evaluation_runner.py
```

## 📈 Outputs

* CSV metrics
* JSON logs
* Attention maps
* RSA heatmaps
* Graph visualizations

## 🔮 Future Work

* Extend to larger LLMs
* Improve reasoning evaluation metrics
* Apply to real-world planning tasks

## 👤 Author

Lakshmi Lahari Satti
AI/ML Engineer | LLM Systems
