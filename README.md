# Retrieval-Augmented Biomedical Language Modeling  
### Domain-Adaptive Fine-Tuning + RAG for Biomedical Summarization and QA

## 📌 Overview
Biomedical documents are long, terminology-dense, and fact-critical. While large language models (LLMs) generate fluent text, they often suffer from **hallucinations**, **poor domain grounding**, and **long-context limitations** when applied directly to biomedical tasks.

This project investigates how **Domain-Adaptive Fine-Tuning (DAPT)** combined with **Retrieval-Augmented Generation (RAG)** can improve factual correctness, semantic relevance, and reliability in biomedical summarization and question answering.

We progressively move from a baseline summarization model to a scalable, retrieval-grounded system evaluated on controlled biomedical benchmarks.

---

## 🎯 Objectives
- Evaluate baseline biomedical summarization performance
- Adapt language models to biomedical domains efficiently
- Reduce hallucinations using external knowledge retrieval
- Study retrieval depth and token-efficiency trade-offs
- Build a reproducible, deployable RAG-based biomedical NLP pipeline

---

## 🧪 Experimental Roadmap

### Phase 1: Baseline Biomedical Summarization (Motivation)
**Model:** T5-small  
**Dataset:** PubMed article–abstract pairs  

This phase establishes a clean baseline and exposes core challenges:
- Limited capacity for biomedical terminology
- High computational cost with weak semantic alignment
- Low factual consistency in generated summaries  

**Outcome:** Identified the need for domain adaptation and retrieval-based grounding.

---

### Phase 2: Scalable Fine-Tuning + Retrieval-Augmented Summarization
**Model:** Phi-3 Mini (3.8B)  
**Techniques:**
- LoRA fine-tuning
- Q4_K_M quantization (llama.cpp)
- Local inference via Ollama
- RAG using Sentence Transformers + FAISS (MedQA corpus)

We evaluated model performance **before and after retrieval augmentation**, observing:
- Improved ROUGE and BERTScore metrics
- Reduced hallucination
- Better factual alignment through retrieved evidence

**Key Insight:** Retrieval contributes more to reliability than fine-tuning alone.

---

### Phase 3: Controlled Biomedical QA with Domain Adaptation + RAG
**Model:** T5-base  
**Dataset:** PubMedQA  

This phase validates the approach on a structured QA benchmark.

**Key Components:**
- **Domain-Adaptive Pre-Training (DAPT):** Span corruption on unlabeled PubMedQA data
- **Pseudo-Label Distillation:** Teacher LLM generates silver labels + explanations
- **RAG-augmented training and inference**

**Results:**
- Accuracy improvement with RAG
- Significant gains on the hardest class ("maybe")
- Higher Macro-F1 score

**Retrieval Study:**  
Focused retrieval (top-k = 1) consistently outperformed higher k values, which introduced noise under token constraints.

---

## 📊 Evaluation Metrics
- ROUGE-1 / ROUGE-2 / ROUGE-L
- BERTScore (semantic similarity)
- Accuracy
- Macro-F1
- Class-wise performance analysis

---

## 🧠 Key Takeaways
- Domain adaptation improves fluency but **retrieval is the primary driver of factual grounding**
- Efficient fine-tuning (LoRA + quantization) enables large-model performance under limited compute
- Focused retrieval outperforms broad retrieval in token-limited settings
- RAG pipelines are essential for trustworthy biomedical NLP systems

---

## 🛠️ Tech Stack
- **Models:** T5-small, T5-base, Phi-3 Mini
- **Fine-Tuning:** LoRA, DAPT
- **Retrieval:** Sentence Transformers, FAISS
- **Deployment:** Ollama, llama.cpp
- **Evaluation:** ROUGE, BERTScore, Accuracy, F1
- **Frameworks:** PyTorch, Hugging Face Transformers

---

## 👤 Author Contributions
**Sanket Deshmukh**
- Designed and implemented Domain-Adaptive Pre-Training (DAPT)
- Built pseudo-label distillation pipeline using teacher LLMs
- Developed end-to-end RAG infrastructure
- Conducted retrieval-depth and performance trade-off analysis
- Led evaluation and experimental analysis

---

## 📦 Data Files

The data files used for this project can be found [here](https://drive.google.com/drive/folders/1L1EPMndxMfQgBtV2_pV3FcqP6NLJF92R?usp=drive_link)

---

## 📁 Project Structure
```
├── data/
│   ├── raw/
│   ├── processed/
│   └── embeddings/
├── models/
│   ├── fine_tuned/
│   └── quantized/
├── retrieval/
│   ├── faiss_index/
│   └── embedding_pipeline.py
├── training/
│   ├── dapt.py
│   ├── finetune_lora.py
│   └── pseudo_labeling.py
├── evaluation/
│   ├── rouge_eval.py
│   ├── bertscore_eval.py
│   └── qa_metrics.py
├── inference/
│   └── rag_inference.py
├── README.md
└── requirements.txt
```

---

## 📌 Future Work
- Larger-scale DAPT on full PubMed Central
- Multi-document retrieval fusion strategies
- Faithfulness-specific metrics (FactCC, QAGS)
- Clinical decision-support extensions

---

## 📄 License
This project is released under the **MIT License** for academic and research use.

---

## ⭐ Acknowledgements
- PubMed / PubMedQA datasets
- Hugging Face Transformers
- FAISS and Sentence Transformers
- Microsoft Phi model family