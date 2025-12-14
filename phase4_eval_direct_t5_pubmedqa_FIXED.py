import re
import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from tqdm.auto import tqdm


def build_source_text(example):
    """
    Build the same input format used during training.
    CRITICAL: Must match phase3 format exactly!
    """
    ctx = " ".join(example["context"]["contexts"])
    q = example["question"]
    la = example["long_answer"] or ""

    example["source_text"] = (
        f"answer question: {q} "
        f"context: {ctx} "
        f"conclusion: {la}"
    )
    return example


def extract_label_from_output(text: str) -> str:
    """
    Extract {yes,no,maybe} from model output.
    Model is trained to output: "label: explanation"
    """
    text_l = text.lower().strip()
    
    # Look for label at start of output
    for cand in ["yes", "no", "maybe"]:
        if text_l.startswith(cand):
            return cand
    
    # Fallback: first occurrence
    for cand in ["yes", "no", "maybe"]:
        if cand in text_l:
            return cand

    # Default to maybe if nothing found
    return "maybe"


def compute_metrics(golds, preds, labels=("yes", "no", "maybe")):
    """Compute accuracy, precision, recall, F1 per class"""
    assert len(golds) == len(preds), f"Mismatch: {len(golds)} golds vs {len(preds)} preds"
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


def main():
    # 1. Load labeled PubMedQA
    print("Loading PubMedQA labeled dataset...")
    dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled")

    # Use train split for evaluation (it's the only labeled split)
    split_name = "train"
    print(f"Using split '{split_name}' for evaluation")
    eval_ds = dataset[split_name]
    print(f"Evaluation examples: {len(eval_ds)}")

    # 2. Build source text
    eval_ds = eval_ds.map(build_source_text, desc="Building prompts")

    # 3. Load fine-tuned T5 model
    model_dir = "./t5_pubmedqa_pseudo_qa"
    
    if not os.path.exists(model_dir):
        raise FileNotFoundError(
            f"Model directory not found: {model_dir}\n"
            f"Please run phase3_finetune_t5_pseudo_qa_FIXED.py first"
        )
    
    print(f"Loading model from {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f"Model loaded on {device}")

    # 4. Extract gold labels
    gold_labels = [g.lower() for g in eval_ds["final_decision"]]
    pred_labels = []

    # 5. Batch generation - FIXED batching logic
    batch_size = 16
    all_sources = eval_ds["source_text"]
    
    print("Running batched generation for direct evaluation...")
    with torch.no_grad():
        for i in tqdm(range(0, len(all_sources), batch_size), desc="Generating"):
            batch_sources = all_sources[i:i + batch_size]
            
            # Tokenize batch
            inputs = tokenizer(
                batch_sources,
                max_length=512,
                truncation=True,
                padding=True,
                return_tensors="pt"
            ).to(device)
            
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

    # 6. Compute metrics
    accuracy, metrics_per_class, cm, labels = compute_metrics(gold_labels, pred_labels)

    # 7. Print results
    print(f"\n{'='*60}")
    print(f"DIRECT T5 (no RAG) on PQA-labeled '{split_name}' split")
    print(f"{'='*60}")
    print(f"Overall accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n")

    print("Per-class metrics:")
    for lab, m in metrics_per_class.items():
        print(
            f"  {lab:>5} | P: {m['precision']:.4f} "
            f"R: {m['recall']:.4f} F1: {m['f1']:.4f} "
            f"(support={m['support']})"
        )

    pretty_print_confusion_matrix(cm, labels)
    
    # Print some example predictions for debugging
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
