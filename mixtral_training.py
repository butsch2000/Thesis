"""
download results:
rsync -avz -e "ssh -i ~/.ssh/id_vast -p 17219" \
  --exclude "pytorch_model.bin" \
  --exclude "pytorch_model.*.bin" \
  --exclude "optimizer.pt" \
  --exclude "scheduler.pt" \
  --exclude "*.safetensors" \
  root@45.135.56.12:/workspace/training/mixtral_outputs/ ./mixtral_outputs_light/
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

# ---------- helpers --------------------------------------------------------

def find_split_files(data_root: Path, ds_id: int) -> Dict[str, Path]:
    tr = list(data_root.glob(f"dataset_{ds_id}_*_train.csv"))
    te = list(data_root.glob(f"dataset_{ds_id}_*_test.csv"))
    if len(tr) != 1 or len(te) != 1:
        raise FileNotFoundError(f"Expect one train/test split for dataset {ds_id}")
    return {"train": tr[0], "test": te[0]}

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    pµ, rµ, fµ, _ = precision_recall_fscore_support(labels, preds,
                                                   average="micro", zero_division=0)
    pM, rM, fM, _ = precision_recall_fscore_support(labels, preds,
                                                   average="macro", zero_division=0)
    return {
        "accuracy": acc,
        "precision_micro": pµ, "recall_micro": rµ, "f1_micro": fµ,
        "precision_macro": pM, "recall_macro": rM, "f1_macro": fM,
    }

# ---------- training loop --------------------------------------------------

def fine_tune_dataset(ds_id: int, data_root: Path, out_root: Path):
    print(f"\n=== DATASET {ds_id} (Mixtral-LoRA) ===")
    paths = find_split_files(data_root, ds_id)

    # 1) load data ----------------------------------------------------------
    raw = load_dataset("csv",
                       data_files={"train": str(paths["train"]),
                                   "test":  str(paths["test"])})
    
    # Replace placeholder for commas
    raw = raw.map(lambda batch: {"text": [text.replace("\\comma", ",") for text in batch["text"]]}, batched=True)

    tokenizer = AutoTokenizer.from_pretrained(
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
        use_fast=True,
        model_max_length=512,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tok(batch):
        return tokenizer(batch["text"], truncation=True)
    
    tokenized = raw.map(tok, batched=True, remove_columns=["text"])
    collator = DataCollatorWithPadding(tokenizer)

    # 2) load model in 4-bit & prepare LoRA -------------------------------
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base_model = AutoModelForSequenceClassification.from_pretrained(
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
        num_labels=5,
        device_map="auto",
        quantization_config=bnb_cfg,
        torch_dtype=torch.bfloat16,
    )

    base_model.gradient_checkpointing_enable()
    base_model = prepare_model_for_kbit_training(base_model)

    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        bias="none",
        lora_dropout=0.05,
        task_type="SEQ_CLS",
    )
    model = get_peft_model(base_model, lora_cfg)
    model.print_trainable_parameters()

    # 3) training args ------------------------------------------------------
    out_dir = out_root / f"dataset_{ds_id}"
    args = TrainingArguments(
        output_dir=str(out_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,     # effective batch ≈16
        learning_rate=2e-4,
        warmup_ratio=0.1,
        weight_decay=0.01,
        fp16=False,                         # doesn't work with 4-bit version
        bf16=True,
        logging_steps=50,
        report_to="wandb",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    metrics = trainer.evaluate()
    trainer.save_metrics("eval", metrics)

    # detailed per-class report
    preds_out = trainer.predict(tokenized["test"])
    preds = np.argmax(preds_out.predictions, axis=-1)
    report = classification_report(
        preds_out.label_ids, preds,
        labels=list(range(5)),
        target_names=[str(i) for i in range(5)],
        zero_division=0, digits=4,
    )
    print(report)
    (out_dir / "classification_report.txt").write_text(report)

# ---------- driver ---------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",   type=str, default="data")
    parser.add_argument("--output_root", type=str, default="mixtral_outputs")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    for i in (1, 2, 3, 4):
        fine_tune_dataset(i, data_root, out_root)

    print("\nAll Mixtral runs finished.")
