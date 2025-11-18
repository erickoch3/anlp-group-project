import os
import argparse
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoConfig,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    GenerationConfig,
)

from semi_rope_marian import SemiRotaryMarianMTModel


def create_preprocess_function(tokenizer):
    def preprocess_function(examples):
        inputs = [ex["de"] for ex in examples["translation"]]
        targets = [ex["en"] for ex in examples["translation"]]
        model_inputs = tokenizer(
            inputs, text_target=targets, return_tensors="pt", padding=True
        )
        return model_inputs

    return preprocess_function


def main():
    parser = argparse.ArgumentParser(
        description="Train Semi-RoPE Marian (RoPE encoder + sinusoidal decoder) on Europarl mini and save to a directory."
    )
    parser.add_argument("--base-model", type=str, default="Helsinki-NLP/opus-mt-de-en", help="Config/tokenizer source")
    parser.add_argument("--output-dir", type=str, default="my-de-en-nmt_semi_rot", help="Where to save the trained model")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--save-total-limit", type=int, default=1)
    parser.add_argument("--report-to", type=str, nargs="*", default=["wandb"], help="Reporting backends e.g. wandb or none via []")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Tokenizer and base config
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    config = AutoConfig.from_pretrained(args.base_model)

    # Fresh model init (random weights) with Semi-RoPE encoder and sinusoidal decoder
    model = SemiRotaryMarianMTModel(config)

    # Load Europarl (Edinburgh) dataset splits
    train_data = load_dataset("EdinburghNLP/europarl-de-en-mini", split="train")
    valid_data = load_dataset("EdinburghNLP/europarl-de-en-mini", split="validation")
    gen_data = load_dataset("EdinburghNLP/europarl-de-en-mini", split="gen_val")

    # Preprocess to token ids
    train_data = train_data.map(create_preprocess_function(tokenizer), batched=True, remove_columns=train_data.column_names)
    valid_proc = valid_data.map(create_preprocess_function(tokenizer), batched=True, remove_columns=valid_data.column_names)
    gen_proc = gen_data.map(create_preprocess_function(tokenizer), batched=True, remove_columns=gen_data.column_names)

    # Training setup
    device_has_cuda = torch.cuda.is_available()
    device_has_mps = getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()

    training_args = Seq2SeqTrainingArguments(
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        auto_find_batch_size=True,
        output_dir=args.output_dir,
        save_total_limit=args.save_total_limit,
        predict_with_generate=False,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps",
        fp16=device_has_cuda,
        fp16_full_eval=device_has_cuda,
        bf16=False,
        group_by_length=True,
        generation_max_length=20,
        report_to=args.report_to,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset={"val": valid_proc, "gen": gen_proc},
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    # Save a generation config aligned with the base model (as in other checkpoints)
    try:
        gen_cfg = GenerationConfig.from_pretrained(args.base_model)
    except Exception:
        gen_cfg = GenerationConfig()
    gen_cfg.save_pretrained(args.output_dir)

    print(f"Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
