import json
from datasets import load_dataset
from tqdm import tqdm
from openai import OpenAI


def build_teacher_prompt(example):
    """
    Ask the teacher LLM to produce a JSON object with:
    - label: one of 'yes', 'no', 'maybe'
    - explanation: short evidence-based text

    We keep the instructions concise and strict to help the model comply.
    """
    contexts = " ".join(example["context"]["contexts"])
    q = example["question"]
    la = example["long_answer"] or ""

    return f"""You are a medical question answering system.

You MUST respond with a single valid JSON object with two fields:
  - "label": one of "yes", "no", or "maybe"
  - "explanation": a short 1-2 sentence evidence-based explanation.

Use ONLY the information from the abstract and the conclusion.

Abstract (without conclusion):
{contexts}

Conclusion section (long answer):
{la}

Question:
{q}
"""


def main():
    # Make sure OPENAI_API_KEY is set in your environment
    # e.g. in zsh: export OPENAI_API_KEY="sk-..."
    client = OpenAI()

    dataset = load_dataset("qiaojin/PubMedQA", "pqa_unlabeled")
    train_ds = dataset["train"]

    # Subsample for cost while testing
    max_examples = 5000  # start small; increase after verifying it works
    train_ds = train_ds.select(range(min(max_examples, len(train_ds))))

    pseudo_labels = []

    for idx, ex in enumerate(tqdm(train_ds, desc="Generating pseudo labels")):
        prompt = build_teacher_prompt(ex)

        try:
            # IMPORTANT: ask for JSON explicitly via response_format
            resp = client.chat.completions.create(
                model="gpt-5-mini",
                messages=[
                    {"role": "system", "content": "You are a medical QA assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content

            # For the first few examples, print the raw content so you can see it
            if idx < 3:
                print("\n--- RAW MODEL OUTPUT ---")
                print(content)
                print("------------------------\n")

            parsed = json.loads(content)

            label = str(parsed.get("label", "")).strip().lower()
            if label not in ["yes", "no", "maybe"]:
                # skip invalid labels
                continue

            explanation = str(parsed.get("explanation", "")).strip()

            pseudo_labels.append({
                "pubid": ex["pubid"],
                "question": ex["question"],
                "contexts": ex["context"]["contexts"],
                "long_answer": ex["long_answer"],
                "label": label,
                "explanation": explanation,
            })

        except Exception as e:
            # If something goes wrong, print *one* example to debug
            if idx < 3:
                print(f"Error on example {idx}: {e}")
            continue

    out_path = "pseudo_labels_pqau_t5.json"
    with open(out_path, "w") as f:
        json.dump(pseudo_labels, f, indent=2)

    print(f"\nSaved {len(pseudo_labels)} pseudo-labeled examples to {out_path}")


if __name__ == "__main__":
    main()
