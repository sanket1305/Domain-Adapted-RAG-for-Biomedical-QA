import re
import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
import faiss
import torch
import numpy as np
from tqdm.auto import tqdm


########################
# Build RAG Corpus
########################

def build_corpus_from_pqa_unlabeled():
    """Build corpus from unlabeled PubMedQA"""
    print("Loading PQA-unlabeled corpus for RAG...")
    ds = load_dataset("qiaojin/PubMedQA", "pqa_unlabeled")["train"]
    
    corpus_texts = []
    corpus_metadata = []

    for ex in ds:
        contexts = " ".join(ex["context"]["contexts"])
        long_ans = ex["long_answer"] or ""
        question = ex["question"]
        
        text = f"{question} {contexts} {long_ans}"
        
        corpus_texts.append(text)
        corpus_metadata.append({
            "pubid": ex["pubid"],
            "contexts": contexts,
            "long_answer": long_ans,
            "question": question,
        })

    print(f"Built corpus with {len(corpus_texts)} documents")
    return corpus_texts, corpus_metadata


########################
# Label extraction & metrics
########################

def extract_label_from_output(text: str) -> str:
    """Extract {yes,no,maybe} from model output."""
    text_l = text.lower().strip()
    
    for cand in ["yes", "no", "maybe"]:
        if text_l.startswith(cand):
            return cand
    
    for cand in ["yes", "no", "maybe"]:
        if cand in text_l:
            return cand

    return "maybe"


def compute_metrics(golds, preds, labels=("yes", "no", "maybe")):
    """Compute accuracy, precision, recall, F1"""
    assert len(golds) == len(preds)
    n = len(golds)

    correct = sum(g == p for g, p in zip(golds, preds))
    accuracy = correct / n if n > 0 else 0.0

    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    cm = [[0 for _ in labels] for _ in labels]

    for g, p in zip(golds, preds):
        if g not in label_to_idx or p not in label_to_idx:
            continue
        gi = label_to_idx[g]
        pi = label_to_idx[p]
        cm[gi][pi] += 1

    metrics_per_class = {}
    for i, lab in enumerate(labels):
        tp = cm[i][i]
        fp = sum(cm[r][i] for r in range(len(labels)) if r != i)
        fn = sum(cm[i][c] for c in range(len(labels)) if c != i)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        metrics_per_class[lab] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(cm[i]),
        }

    return accuracy, metrics_per_class, cm, labels


def pretty_print_confusion_matrix(cm, labels):
    """Print confusion matrix"""
    print("\nConfusion Matrix (rows = GOLD, cols = PRED):")
    header = "gold\\pred".ljust(10) + " ".join(l.rjust(8) for l in labels)
    print(header)
    for i, lab in enumerate(labels):
        row_counts = " ".join(str(cm[i][j]).rjust(8) for j in range(len(labels)))
        print(lab.ljust(10) + row_counts)


########################
# Main RAG evaluation - SIMPLE APPROACH
########################

def main():
    # Configuration
    USE_RAG = True  # Set to False to test without RAG
    k = 1  # Use ONLY 1 best retrieved document
    
    # 1. Build retrieval corpus
    corpus_texts, corpus_metadata = build_corpus_from_pqa_unlabeled()

    # 2. Load embedding model
    print("Loading sentence embedding model...")
    embed_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embedder = SentenceTransformer(embed_model_name)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedder = embedder.to(device)
    print(f"Embedder on {device}")

    # 3. Build FAISS index
    print("Computing corpus embeddings...")
    corpus_embs = embedder.encode(
        corpus_texts,
        batch_size=128,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
        device=device,
    )

    dim = corpus_embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(corpus_embs.astype(np.float32))
    print(f"FAISS index built with {index.ntotal} vectors")

    # 4. Load evaluation data
    print("Loading PQA-labeled split for evaluation...")
    dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled")
    split_name = "train"
    eval_ds = dataset[split_name]
    print(f"Evaluation examples: {len(eval_ds)}")

    # 5. Load model
    model_dir = "./t5_pubmedqa_pseudo_qa"
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    print(f"Loading T5 model from {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    
    model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(model_device)
    model.eval()
    print(f"Model loaded on {model_device}")

    # 6. Build prompts - SIMPLIFIED APPROACH
    gold_labels = []
    source_prompts = []

    print(f"Building prompts (RAG={'ON' if USE_RAG else 'OFF'}, k={k if USE_RAG else 0})...")
    
    for ex in tqdm(eval_ds, desc="Building prompts"):
        q = ex["question"]
        gold = ex["final_decision"].lower()
        
        # Original context
        orig_context = " ".join(ex["context"]["contexts"])
        orig_conclusion = ex["long_answer"] or ""
        
        if USE_RAG:
            # Simple retrieval: just use the question
            q_emb = embedder.encode(
                [q], 
                normalize_embeddings=True,
                convert_to_numpy=True,
                device=device,
            )
            D, I = index.search(q_emb.astype(np.float32), k)
            
            # Get top-1 retrieved doc
            if I[0][0] != -1:  # Valid index
                meta = corpus_metadata[I[0][0]]
                retrieved_ctx = meta['contexts'][:300]  # First 300 chars
                
                # Append retrieved as additional evidence
                source = (
                    f"answer question: {q} "
                    f"context: {orig_context} "
                    f"conclusion: {orig_conclusion} "
                    f"additional evidence: {retrieved_ctx}"
                )
            else:
                # Fallback to no RAG
                source = (
                    f"answer question: {q} "
                    f"context: {orig_context} "
                    f"conclusion: {orig_conclusion}"
                )
        else:
            # No RAG - just original
            source = (
                f"answer question: {q} "
                f"context: {orig_context} "
                f"conclusion: {orig_conclusion}"
            )

        source_prompts.append(source)
        gold_labels.append(gold)

    # 7. Batch generation
    print("Running T5 generation...")
    pred_labels = []
    batch_size = 16  # Larger batch since we have less context
    
    with torch.no_grad():
        for i in tqdm(range(0, len(source_prompts), batch_size), desc="Generating"):
            batch_sources = source_prompts[i:i + batch_size]
            
            inputs = tokenizer(
                batch_sources,
                max_length=512,  # Standard length
                truncation=True,
                padding=True,
                return_tensors="pt"
            ).to(model_device)
            
            outputs = model.generate(
                **inputs,
                max_new_tokens=64,
                num_beams=4,
                early_stopping=True,
            )
            
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            for text in decoded:
                pred_labels.append(extract_label_from_output(text))

    # 8. Compute metrics
    accuracy, metrics_per_class, cm, labels = compute_metrics(gold_labels, pred_labels)

    # 9. Print results
    print(f"\n{'='*60}")
    print(f"{'RAG-based' if USE_RAG else 'Direct'} T5 on PQA-labeled '{split_name}' split")
    print(f"{'='*60}")
    if USE_RAG:
        print(f"Retrieval: top-{k} document(s)")
    print(f"Overall accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n")

    print("Per-class metrics:")
    for lab, m in metrics_per_class.items():
        print(
            f"  {lab:>5} | P: {m['precision']:.4f} "
            f"R: {m['recall']:.4f} F1: {m['f1']:.4f} "
            f"(support={m['support']})"
        )

    pretty_print_confusion_matrix(cm, labels)
    
    # Sample predictions
    print(f"\n{'='*60}")
    print("Sample predictions (first 5 examples):")
    print(f"{'='*60}")
    for i in range(min(5, len(gold_labels))):
        print(f"\nExample {i+1}:")
        print(f"  Question: {eval_ds[i]['question'][:80]}...")
        print(f"  Gold: {gold_labels[i]}")
        print(f"  Pred: {pred_labels[i]}")
        print(f"  {'✓' if gold_labels[i] == pred_labels[i] else '✗'}")


if __name__ == "__main__":
    main()
