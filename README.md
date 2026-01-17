🏥 Medical Visual Question Answering (Med-VQA)
Comparative Analysis: CNN-LSTM Baseline vs. BLIP-2 VLM

This repository contains the implementation and evaluation of two different deep learning approaches for Medical Visual Question Answering on the VQA-RAD dataset. This project was developed as part of the WOA7015 Advanced Machine Learning course at Universiti Malaya.

📌 Project Overview
The challenge of Med-VQA is to interpret complex medical images (X-rays, CT, MRI) and provide accurate answers to clinical natural language questions. We compare a "Specialist" approach (trained from scratch) against a "Generalist" approach (Large Vision-Language Model).

Key Objectives:

Implement a CNN-LSTM architecture with a visual attention mechanism as a performance baseline.

Evaluate the Zero-Shot capabilities of the BLIP-2 Vision-Language Model (VLM).

Analyze performance across Closed-ended (Yes/No, Modality) and Open-ended (Descriptive) question types.

🏗️ Architectures
1. CNN-LSTM Baseline (The Specialist)

Visual Encoder: ResNet-50 (Pre-trained on ImageNet) for feature extraction.

Linguistic Encoder: 2-layer Bidirectional LSTM for processing clinical questions.

Fusion: Visual Attention Mechanism that weights image regions based on the question context.

Approach: Supervised learning (classification-based) trained specifically on the VQA-RAD training split.

2. BLIP-2 (The Generalist)

Model: Salesforce/blip2-flan-t5-xl.

Approach: Zero-Shot Inference (No medical fine-tuning).

Optimizations: * Question-type aware prompting.

Aggressive answer normalization.

Flexible Matching: A custom evaluation metric to handle natural language variations (e.g., "The image shows a CT" vs. "CT").

📊 Experimental Results
Metric	CNN-LSTM (Baseline)	BLIP-2 (Zero-Shot)
Overall Accuracy	34.2%	41.2%
Closed-ended Accuracy	52.9%	43.8%
Open-ended Accuracy	0.6%	27.8%
Inference Mode	Trained/Classification	Zero-Shot/Generative
Key Finding: The CNN-LSTM specialist excels at structured, closed-ended patterns it was trained on, whereas BLIP-2 demonstrates superior reasoning for descriptive, open-ended queries.

🛠️ Data Integrity & Leakage Prevention
To ensure valid results, we implemented image-level GroupShuffleSplit. This prevents data leakage by ensuring that if an image is in the test set, no questions associated with that same image appear in the training set.

🚀 Getting Started
Prerequisites

Bash
pip install torch torchvision transformers datasets scikit-learn nltk
Files

CNN_LSTM_MedVQA-vF.ipynb: Implementation of the baseline model, training loops, and attention visualization.

BLIP2_ZeroShot_MedVQA_vF.ipynb: Implementation of the VLM evaluation with optimized prompting and flexible matching logic.
