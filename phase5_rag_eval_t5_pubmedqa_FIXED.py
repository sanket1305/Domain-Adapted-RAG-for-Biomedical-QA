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
    """
    Build a corpus of PubMed abstracts from unlabeled PubMedQA.
    Each corpus document contains: contexts + long_answer + question
    This provides richer retrieval signal.
    """
    print("Loading PQA-unlabeled corpus for RAG...")
    ds = load_dataset("qiaojin/PubMedQA", "pqa_unlabeled")["train"]
    
    corpus_texts = []
    corpus_metadata = []

    for ex in ds:
        # Combine all text for better retrieval
        contexts = " ".join(ex["context"]["contexts"])
        long_ans = ex["long_answer"] or ""
        question = ex["question"]
        
        # Full document text
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
    
    # Look for label at start
    for cand in ["yes", "no", "maybe"]:
        if text_l.startswith(cand):
            return cand
    
    # Fallback
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
# Main RAG evaluation
########################

def main():
    # 1. Build retrieval corpus from unlabeled PubMedQA
    corpus_texts, corpus_metadata = build_corpus_from_pqa_unlabeled()

    # 2. Load sentence embedding model
    print("Loading sentence embedding model...")
    embed_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embedder = SentenceTransformer(embed_model_name)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedder = embedder.to(device)
    print(f"Embedder on {device}")

    # 3. Compute corpus embeddings + build FAISS index
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
    index = faiss.IndexFlatIP(dim)  # Inner product on normalized = cosine
    index.add(corpus_embs.astype(np.float32))
    print(f"FAISS index built with {index.ntotal} vectors")

    # 4. Load labeled PubMedQA evaluation split
    print("Loading PQA-labeled split for RAG evaluation...")
    dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled")
    split_name = "train"
    print(f"Using split '{split_name}' for RAG evaluation")
    eval_ds = dataset[split_name]
    print(f"Evaluation examples: {len(eval_ds)}")

    # 5. Load fine-tuned T5 QA model
    model_dir = "./t5_pubmedqa_pseudo_qa"
    
    if not os.path.exists(model_dir):
        raise FileNotFoundError(
            f"Model directory not found: {model_dir}\n"
            f"Please run phase3_finetune_t5_pseudo_qa_FIXED.py first"
        )
    
    print(f"Loading T5 model from {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    
    model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(model_device)
    model.eval()
    print(f"Model loaded on {model_device}")

    # 6. RAG: build prompts via retrieval
    k = 7  # Number of retrieved docs (reduced from 5 to fit in context better)
    gold_labels = []
    source_prompts = []

    print("Building RAG prompts (retrieval step)...")
    for ex in tqdm(eval_ds, desc="Retrieval"):
        q = ex["question"]
        gold = ex["final_decision"].lower()
        
        # Also include the original context in query for better retrieval
        orig_context = " ".join(ex["context"]["contexts"])
        query_text = f"{q} {orig_context}"

        # Retrieve similar documents
        q_emb = embedder.encode(
            [query_text], 
            normalize_embeddings=True,
            convert_to_numpy=True,
            device=device,
        )
        D, I = index.search(q_emb.astype(np.float32), k)

        # Build retrieved context
        retrieved_parts = []
        for j, idx in enumerate(I[0]):
            meta = corpus_metadata[idx]
            # Use contexts + long_answer from retrieved docs
            retrieved_parts.append(
                f"[Retrieved {j+1}] {meta['contexts']} {meta['long_answer']}"
            )
        
        retrieved_context = " ".join(retrieved_parts)

        # CRITICAL: Use original context too (hybrid approach)
        # This combines the test example's context with retrieved contexts
        source = (
            f"answer question: {q} "
            f"context: {orig_context} "
            f"retrieved context: {retrieved_context}"
        )

        source_prompts.append(source)
        gold_labels.append(gold)

    # 7. Batch generation
    print("Running T5 generation in batches...")
    pred_labels = []
    batch_size = 8
    
    with torch.no_grad():
        for i in tqdm(range(0, len(source_prompts), batch_size), desc="Generating"):
            batch_sources = source_prompts[i:i + batch_size]
            
            # Tokenize
            inputs = tokenizer(
                batch_sources,
                max_length=512,
                truncation=True,
                padding=True,
                return_tensors="pt"
            ).to(model_device)
            
            # Generate
            outputs = model.generate(
                **inputs,
                max_new_tokens=64,
                num_beams=4,
                early_stopping=True,
            )
            
            # Decode
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            # Extract labels
            for text in decoded:
                pred_labels.append(extract_label_from_output(text))

    # 8. Compute metrics
    accuracy, metrics_per_class, cm, labels = compute_metrics(gold_labels, pred_labels)

    # 9. Print results
    print(f"\n{'='*60}")
    print(f"RAG-based T5 on PQA-labeled '{split_name}' split")
    print(f"{'='*60}")
    print(f"Retrieval: top-{k} documents")
    print(f"Overall accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n")

    print("Per-class metrics:")
    for lab, m in metrics_per_class.items():
        print(
            f"  {lab:>5} | P: {m['precision']:.4f} "
            f"R: {m['recall']:.4f} F1: {m['f1']:.4f} "
            f"(support={m['support']})"
        )

    pretty_print_confusion_matrix(cm, labels)
    
    # Print some examples
    print(f"\n{'='*60}")
    print("Sample RAG predictions (first 5 examples):")
    print(f"{'='*60}")
    for i in range(min(5, len(gold_labels))):
        print(f"\nExample {i+1}:")
        print(f"  Question: {eval_ds[i]['question'][:80]}...")
        print(f"  Gold: {gold_labels[i]}")
        print(f"  Pred: {pred_labels[i]}")
        print(f"  {'✓' if gold_labels[i] == pred_labels[i] else '✗'}")


if __name__ == "__main__":
    main()
