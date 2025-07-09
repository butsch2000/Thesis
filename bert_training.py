"""
download results:
rsync -avz -e "ssh -i ~/.ssh/id_vast -p 19465" \
  --exclude "pytorch_model.bin" \
  --exclude "pytorch_model.*.bin" \
  --exclude "optimizer.pt" \
  --exclude "scheduler.pt" \
  --exclude "*.safetensors" \
  root@198.145.126.233:/workspace/training/balanced_bert_outputs/ ./balanced_bert_outputs_light/
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
)


# ---------------- utility --------------------------------------------------

def find_split_files(data_root: Path, ds_id: int) -> Dict[str, Path]:
    """Return {'train': Path, 'test': Path} for dataset <ds_id>."""
    train_candidate = list(data_root.glob(f"dataset_{ds_id}_*_train.csv"))
    test_candidate = list(data_root.glob(f"dataset_{ds_id}_*_test.csv"))
    if len(train_candidate) != 1 or len(test_candidate) != 1:
        raise FileNotFoundError(
            f"Expected exactly one train/test CSV for dataset {ds_id} in {data_root}"
        )
    return {"train": train_candidate[0], "test": test_candidate[0]}


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, preds)
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        labels, preds, average="micro", zero_division=0
    )
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )

    return {
        "accuracy": acc,
        "precision_micro": precision_micro,
        "recall_micro": recall_micro,
        "f1_micro": f1_micro,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
    }


# ---------------- main -----------------------------------------------------

def fine_tune_dataset(ds_id: int, data_root: Path, output_root: Path, device: str):
    print(f"\n=== Dataset {ds_id} ===")
    paths = find_split_files(data_root, ds_id)

    # Load dataset with the 🤗 Datasets library
    raw = load_dataset(
        "csv",
        data_files={"train": str(paths["train"]), "test": str(paths["test"])},
    )
    # Replace placeholder for commas
    raw = raw.map(lambda batch: {"text": [text.replace("\\comma", ",") for text in batch["text"]]}, batched=True)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-german-cased")

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True)

    tokenized = raw.map(tokenize, batched=True, remove_columns=["text"])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-german-cased", num_labels=5
    ).to(device)

    training_args = TrainingArguments(
        output_dir=str(output_root / f"dataset_{ds_id}"),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        fp16=torch.cuda.is_available(),
        report_to="wandb",
        logging_steps=50,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # Final evaluation
    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

    # Detailed per‑class report
    preds_output = trainer.predict(tokenized["test"])
    preds = np.argmax(preds_output.predictions, axis=-1)
    report = classification_report(
        preds_output.label_ids,
        preds,
        labels=list(range(5)),
        digits=4,
        zero_division=0,
        target_names=[str(i) for i in range(5)],
    )
    print(report)
    (output_root / f"dataset_{ds_id}" / "classification_report.txt").write_text(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data", help="Directory with CSV splits")
    parser.add_argument("--output_root", type=str, default="bert_outputs", help="Where to store fine‑tuned models & reports")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    for ds_id in (1, 2, 3, 4):
        fine_tune_dataset(ds_id, data_root, output_root, device)

    print("\nAll datasets processed.")


# final run stats:
# https://wandb.ai/benutsch2000-technical-university-of-berlin/huggingface/runs/40r1l143?nw=nwuserbenutsch2000