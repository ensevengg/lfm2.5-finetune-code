"""
Phase 2: LoRA fine-tune LFM2.5-1.2B-Thinking on KodCode
Designed for Lightning.ai L40S Studio (48 GB VRAM)

Usage:
    python train_lfm25_lora.py                                    # fresh run, Hub push
    python train_lfm25_lora.py --resume                           # resume latest checkpoint
    python train_lfm25_lora.py --dataset enseven/kodcode-lfm2.5   # load from Hub dataset
    python train_lfm25_lora.py --dataset ./hf_dataset              # load from local disk
    python train_lfm25_lora.py --no-hub                           # skip Hub push
"""

import argparse
import os
import re
from glob import glob

os.environ["TOKENIZERS_PARALLELISM"] = "true"


def check_gpu():
    import torch
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available — training will be slow or fail", flush=True)
        return
    name = torch.cuda.get_device_name()
    mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU: {name} | VRAM: {mem:.0f} GB | CUDA: {torch.version.cuda}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument("--dataset", type=str, default="enseven/kodcode-lfm2.5",
                        help="HF dataset repo or local path")
    parser.add_argument("--dataset-text-field", type=str, default="text",
                        help="Column name containing formatted text")
    parser.add_argument("--max-seq-length", type=int, default=4096,
                        help="Truncate/pack sequences to this length")
    parser.add_argument("--packing", action="store_true", default=True,
                        help="Pack multiple short examples into one sequence")
    parser.add_argument("--no-packing", action="store_false", dest="packing")

    # Model
    parser.add_argument("--model-name", type=str,
                        default="LiquidAI/LFM2.5-1.2B-Thinking")
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.1)

    # Training
    parser.add_argument("--output-dir", type=str, default="./lfm25-kodcode-lora")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-scheduler-type", type=str, default="cosine",
                        help="LR scheduler (cosine, linear, constant_with_warmup)")
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--eval-split", type=float, default=0.01,
                        help="Fraction of training data to hold out for eval")
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=250)
    parser.add_argument("--save-total-limit", type=int, default=3,
                        help="Keep at most N checkpoints on disk")
    parser.add_argument("--max-grad-norm", type=float, default=1.0,
                        help="Gradient clipping norm")
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--no-bf16", action="store_false", dest="bf16")
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True)
    parser.add_argument("--no-gradient-checkpointing", action="store_false",
                        dest="gradient_checkpointing")

    # Resume
    parser.add_argument("--resume", action="store_true", default=False,
                        help="Resume from latest checkpoint in output-dir")
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Explicit path to checkpoint directory")

    # Hub
    parser.add_argument("--hub-output", type=str, default="enseven/lfm-2.5-think-code",
                        help="HF repo to push model")
    parser.add_argument("--hub-token", type=str, default=None,
                        help="HF write token for model push")
    parser.add_argument("--no-hub", action="store_true", default=False,
                        help="Skip pushing to Hub even if --hub-output is set")

    # Tracking
    parser.add_argument("--wandb-project", type=str, default="lfm25-kodcode",
                        help="W&B project name (empty string to disable)")

    return parser.parse_args()


def main():
    args = parse_args()
    check_gpu()

    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
    )
    from trl import SFTTrainer
    from peft import LoraConfig
    from datasets import load_dataset, load_from_disk

    # ── Load dataset ────────────────────────────────────────────────────
    print(f"Loading dataset from {args.dataset} ...", flush=True)
    if "/" in args.dataset and not os.path.isdir(args.dataset):
        dataset = load_dataset(args.dataset, split="train")
    else:
        dataset = load_from_disk(args.dataset)

    if args.eval_split > 0:
        split = dataset.train_test_split(test_size=args.eval_split, seed=42)
        train_dataset = split["train"]
        eval_dataset = split["test"]
        print(f"  Train: {len(train_dataset)} | Eval: {len(eval_dataset)}", flush=True)
    else:
        train_dataset = dataset
        eval_dataset = None
        print(f"  {len(train_dataset)} rows (no eval split)", flush=True)

    # ── Load model + tokenizer ──────────────────────────────────────────
    print(f"Loading model {args.model_name} ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    print(f"  Parameters: {model.num_parameters():,}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── LoRA config ─────────────────────────────────────────────────────
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=[
            "in_proj", "out_proj",  # ShortConv projection layers
            "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
            "w1", "w2", "w3",  # FFN (all layers)
        ],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    print(
        f"LoRA: r={args.lora_r}, alpha={args.lora_alpha}, "
        f"modules={lora_config.target_modules}",
        flush=True,
    )

    # ── Reporting (W&B) ────────────────────────────────────────────────
    if args.wandb_project:
        try:
            import wandb
            wandb.init(project=args.wandb_project)
            report_to = ["wandb"]
        except Exception as e:
            print(f"W&B skipped: {e}", flush=True)
            report_to = ["none"]
    else:
        report_to = ["none"]

    # ── Training args ───────────────────────────────────────────────────
    push_hub = args.hub_output and not args.no_hub
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.epochs,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=args.save_steps,
        max_grad_norm=args.max_grad_norm,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to=report_to,
        push_to_hub=push_hub,
        hub_model_id=args.hub_output,
        hub_token=args.hub_token,
        hub_strategy="every_save",
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        dataloader_num_workers=4,
    )

    effective_batch = args.batch_size * args.grad_accum
    steps_per_epoch = len(train_dataset) // effective_batch
    print(
        f"Training: batch={args.batch_size}, grad_accum={args.grad_accum}, "
        f"effective_batch={effective_batch}, steps/epoch≈{steps_per_epoch}",
        flush=True,
    )

    # ── Trainer ─────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field=args.dataset_text_field,
        max_seq_length=args.max_seq_length,
        packing=args.packing,
        peft_config=lora_config,
    )

    # ── Resume checkpoint ───────────────────────────────────────────────
    resume_checkpoint = None
    if args.resume_from:
        resume_checkpoint = args.resume_from
        print(f"Resuming from explicit checkpoint: {resume_checkpoint}", flush=True)
    elif args.resume:
        checkpoints = sorted(glob(os.path.join(args.output_dir, "checkpoint-*")),
                             key=lambda x: int(re.search(r"checkpoint-(\d+)", x).group(1)))
        if checkpoints:
            resume_checkpoint = checkpoints[-1]
            print(f"Resuming from latest checkpoint: {resume_checkpoint}", flush=True)
        else:
            print("No checkpoints found in output-dir, starting fresh.", flush=True)

    # ── Train ───────────────────────────────────────────────────────────
    print("\nStarting training ...", flush=True)
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    print("Training complete!", flush=True)

    # ── Save ────────────────────────────────────────────────────────────
    print(f"Saving LoRA adapter to {args.output_dir} ...", flush=True)
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    print("Done!", flush=True)


if __name__ == "__main__":
    main()
