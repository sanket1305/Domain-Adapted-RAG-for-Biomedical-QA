import random
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)
import torch
import numpy as np


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_raw_text(example):
    """
    Build a single text field from question + context + long_answer.
    This makes the DAPT objective 'see' the QA structure + biomedical language.
    """
    contexts = " ".join(example["context"]["contexts"])
    q = example["question"]
    la = example["long_answer"] or ""

    # Consistent format with evaluation phases
    text = f"Question: {q}\nContext: {contexts}\nConclusion: {la}"
    example["text"] = text
    return example


def span_corruption(text, mask_ratio=0.15, mean_span_length=3):
    """
    T5-style span corruption for denoising objective.
    Replaces random spans with sentinel tokens.
    """
    tokens = text.split()
    if len(tokens) == 0:
        return text, text
    
    n_tokens = len(tokens)
    n_mask = max(1, int(n_tokens * mask_ratio))
    
    # Generate random spans to mask
    mask_indices = set()
    while len(mask_indices) < n_mask:
        start = random.randint(0, n_tokens - 1)
        span_length = np.random.poisson(mean_span_length)
        span_length = max(1, min(span_length, n_tokens - start))
        for i in range(start, start + span_length):
            mask_indices.add(i)
    
    # Build corrupted input and target
    corrupted = []
    target = []
    sentinel_id = 0
    i = 0
    
    while i < n_tokens:
        if i in mask_indices:
            # Start of a masked span
            span_tokens = []
            while i < n_tokens and i in mask_indices:
                span_tokens.append(tokens[i])
                i += 1
            
            sentinel = f"<extra_id_{sentinel_id}>"
            corrupted.append(sentinel)
            target.append(sentinel)
            target.extend(span_tokens)
            sentinel_id += 1
        else:
            corrupted.append(tokens[i])
            i += 1
    
    # Add final sentinel to target
    target.append(f"<extra_id_{sentinel_id}>")
    
    return " ".join(corrupted), " ".join(target)


def main():
    set_seed(42)

    # 1. Load PQA-U (unlabeled PubMedQA)
    print("Loading PubMedQA unlabeled dataset...")
    dataset = load_dataset("qiaojin/PubMedQA", "pqa_unlabeled")
    train_ds = dataset["train"]
    print(f"Loaded {len(train_ds)} unlabeled examples")

    # 2. Build "text" field
    train_ds = train_ds.map(build_raw_text, desc="Building text field")

    # 3. Choose base T5 model
    model_name = "t5-base"  # More reliable than t5-base-lm-adapt
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    max_input_len = 512
    max_target_len = 128

    def preprocess_dapt(examples):
        """Process in batches for efficiency"""
        batch_corrupted = []
        batch_targets = []
        
        for text in examples["text"]:
            corrupted, target = span_corruption(text, mask_ratio=0.15)
            batch_corrupted.append(corrupted)
            batch_targets.append(target)

        # Encoder input: corrupted text
        model_inputs = tokenizer(
            batch_corrupted,
            max_length=max_input_len,
            truncation=True,
            padding=False,  # DataCollator will handle padding
        )

        # Decoder target: spans to predict
        labels = tokenizer(
            batch_targets,
            max_length=max_target_len,
            truncation=True,
            padding=False,
        )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    print("Tokenizing dataset...")
    tokenized_train = train_ds.map(
        preprocess_dapt,
        batched=True,
        batch_size=1000,
        remove_columns=train_ds.column_names,
        desc="Tokenizing DAPT",
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer, 
        model=model,
        padding=True,
    )

    training_args = TrainingArguments(
        output_dir="./t5_pubmedqa_dapt",
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        num_train_epochs=3,
        warmup_steps=500,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        logging_steps=100,
        save_steps=2000,
        save_total_limit=3,
        report_to="none",
        dataloader_num_workers=4,
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print("Starting DAPT training...")
    trainer.train()
    
    print("Saving model...")
    trainer.save_model("./t5_pubmedqa_dapt")
    tokenizer.save_pretrained("./t5_pubmedqa_dapt")
    print("DAPT phase complete!")


if __name__ == "__main__":
    main()
