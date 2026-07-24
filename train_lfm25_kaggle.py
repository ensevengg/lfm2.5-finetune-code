"""
LoRA fine-tune LFM2.5-1.2B-Thinking on Kaggle (T4 GPU, free tier)
Uses QLoRA (4-bit) + fp16 compute with no AMP scaler (avoids T4 bf16 kernel bug).
Resumes from Hub checkpoint saved by the Lightning run.

Usage (Kaggle Notebook — T4 x2, drag in only the .py file):
    !pip install --upgrade transformers bitsandbytes trl peft datasets accelerate sentencepiece wandb
    !torchrun --nproc_per_node=2 train_lfm25_kaggle.py

Usage (resume after disconnect):
    !torchrun --nproc_per_node=2 train_lfm25_kaggle.py \\
        --resume-from-hub-checkpoint enseven/lfm-2.5-think-code
"""

import argparse
import os
import re
from glob import glob

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ["ACCELERATE_MIXED_PRECISION"] = "no"

LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))


ASSISTANT_HEADER = "<|im_start|>assistant\n"
BOS_TOKEN = "<|startoftext|>"


def check_gpu():
    import torch
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available — training will be slow or fail", flush=True)
        return
    name = torch.cuda.get_device_name()
    mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU: {name} | VRAM: {mem:.0f} GB | CUDA: {torch.version.cuda}", flush=True)


def setup_kaggle_secrets():
    try:
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        token = user_secrets.get_secret("HF_TOKEN")
        if token:
            os.environ["HF_TOKEN"] = token
        wandb_key = user_secrets.get_secret("WANDB_API_KEY")
        if wandb_key:
            os.environ["WANDB_API_KEY"] = wandb_key
    except ImportError:
        pass


def to_prompt_completion(example, index, text_field):
    """Convert the legacy flat chat string into TRL's loss-masked dataset format."""
    text = example[text_field]
    if not isinstance(text, str):
        raise ValueError(f"Row {index} has a non-string {text_field!r} value")

    prompt, marker, completion = text.rpartition(ASSISTANT_HEADER)
    if not marker or not prompt or not completion:
        raise ValueError(
            f"Row {index} does not match the expected LFM chat format. "
            f"Expected one final {ASSISTANT_HEADER!r} marker."
        )
    # SFTTrainer adds the tokenizer's BOS and EOS itself. The legacy format
    # already contains both, so retaining them would duplicate those tokens.
    if prompt.startswith(BOS_TOKEN):
        prompt = prompt[len(BOS_TOKEN):]
    completion = completion.rstrip()
    return {"prompt": prompt + marker, "completion": completion}


def fits_within_context(example, tokenizer, max_length):
    # Match SFTTrainer's standard prompt-completion tokenization before packing.
    input_ids = tokenizer(example["prompt"] + example["completion"])["input_ids"]
    return len(input_ids) <= max_length


def parse_args():
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument("--dataset", type=str, default="enseven/kodcode-lfm2.5",
                        help="HF dataset repo or local path")
    parser.add_argument("--dataset-text-field", type=str, default="text",
                        help="Column name containing formatted text")
    parser.add_argument("--max-seq-length", type=int, default=2048,
                        help="Truncate/pack sequences (2048 for T4 memory)")
    parser.add_argument("--packing", action="store_true", default=False,
                        help="Pack examples only when using a verified FlashAttention backend")
    parser.add_argument("--no-packing", action="store_false", dest="packing")
    parser.add_argument("--filter-overlong", action="store_true", default=True,
                        help="Drop rows that would truncate any completion token")
    parser.add_argument("--no-filter-overlong", action="store_false", dest="filter_overlong")
    parser.add_argument("--dataset-num-proc", type=int, default=1,
                        help="Workers for deterministic dataset preprocessing")

    # Model
    parser.add_argument("--model-name", type=str,
                        default="LiquidAI/LFM2.5-1.2B-Thinking")
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.1)

    # Training
    parser.add_argument("--output-dir", type=str, default="./lfm25-kodcode-lora")
    parser.add_argument("--batch-size", type=int, default=2,
                        help="T4 has 15 GB VRAM, keep this low")
    parser.add_argument("--eval-batch-size", type=int, default=2,
                        help="Keep evaluation within the same T4 VRAM budget")
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-scheduler-type", type=str, default="cosine")
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--eval-split", type=float, default=0.01)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=250)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--max-grad-norm", type=float, default=0.0,
                        help="Set to 0 to disable (avoids GradScaler bf16 bug on T4)")
    parser.add_argument("--optim", type=str, default="adamw_torch",
                        help="Optimizer (adamw_torch avoids bf16 gradient scaler issues on T4)")
    parser.add_argument("--seed", type=int, default=42)

    # Resume
    parser.add_argument("--resume", action="store_true", default=False,
                        help="Resume from local checkpoint (same machine)")
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Local path to checkpoint")
    parser.add_argument("--resume-from-hub", type=str, default=None,
                        help="HF repo ID to continue adapter weights from (optimizer starts fresh)")
    parser.add_argument("--resume-from-hub-checkpoint", type=str, default=None,
                        help="'namespace/repo' or 'namespace/repo/checkpoint-NNN' — full resume (weights + optimizer + step)")

    # Hub
    parser.add_argument("--hub-output", type=str, default="enseven/lfm-2.5-think-code",
                        help="HF repo to push model")
    parser.add_argument("--hub-token", type=str, default=None,
                        help="HF write token (falls back to HF_TOKEN env var)")
    parser.add_argument("--no-hub", action="store_true", default=False)

    # Tracking
    parser.add_argument("--wandb-project", type=str, default="lfm25-kodcode",
                        help="W&B project name (empty string to disable)")

    return parser.parse_args()


def main():
    args = parse_args()
    check_gpu()
    setup_kaggle_secrets()

    if args.hub_token is None:
        args.hub_token = os.environ.get("HF_TOKEN")
    if args.hub_token is None and args.hub_output and not args.no_hub:
        print("WARNING: --hub-output is set but no HF_TOKEN found. Push will fail.", flush=True)

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    print(f"Rank {LOCAL_RANK}/{world_size - 1} — device CUDA:{LOCAL_RANK}", flush=True)

    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
    from trl import SFTConfig, SFTTrainer
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from datasets import load_dataset, load_from_disk

    resume_sources = sum(
        bool(value)
        for value in (args.resume, args.resume_from, args.resume_from_hub, args.resume_from_hub_checkpoint)
    )
    if resume_sources > 1:
        raise ValueError("Choose only one of --resume, --resume-from, --resume-from-hub, or --resume-from-hub-checkpoint")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
    )
    if tokenizer.eos_token is None:
        raise ValueError("The tokenizer must define an EOS token for SFT.")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ── Load dataset ────────────────────────────────────────────────────
    print(f"Loading dataset from {args.dataset} ...", flush=True)
    if "/" in args.dataset and not os.path.isdir(args.dataset):
        dataset = load_dataset(args.dataset, split="train")
    else:
        dataset = load_from_disk(args.dataset)

    if {"prompt", "completion"}.issubset(dataset.column_names):
        print("  Using prompt/completion fields with completion-only loss.", flush=True)
    elif args.dataset_text_field in dataset.column_names:
        dataset = dataset.map(
            to_prompt_completion,
            with_indices=True,
            fn_kwargs={"text_field": args.dataset_text_field},
            remove_columns=dataset.column_names,
            num_proc=args.dataset_num_proc,
            desc="Converting legacy chat strings to prompt/completion",
        )
        print("  Converted legacy chat strings to prompt/completion fields.", flush=True)
    else:
        raise ValueError(
            f"Dataset must contain prompt/completion or {args.dataset_text_field!r}; "
            f"found {dataset.column_names}"
        )

    if args.filter_overlong:
        before = len(dataset)
        dataset = dataset.filter(
            fits_within_context,
            fn_kwargs={"tokenizer": tokenizer, "max_length": args.max_seq_length},
            num_proc=args.dataset_num_proc,
            desc=f"Filtering rows longer than {args.max_seq_length} tokens",
        )
        if not len(dataset):
            raise ValueError("All rows exceed --max-seq-length. Increase it or inspect the dataset.")
        print(f"  Kept {len(dataset):,}/{before:,} rows without completion truncation.", flush=True)

    if args.eval_split > 0:
        split = dataset.train_test_split(test_size=args.eval_split, seed=args.seed)
        train_dataset = split["train"]
        eval_dataset = split["test"]
        print(f"  Train: {len(train_dataset)} | Eval: {len(eval_dataset)}", flush=True)
    else:
        train_dataset = dataset
        eval_dataset = None
        print(f"  {len(train_dataset)} rows (no eval split)", flush=True)

    # ── Load model with QLoRA (4-bit + fp16 compute, no AMP scaler) ─────
    print(f"Loading model {args.model_name} with QLoRA (4-bit, no AMP) ...", flush=True)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        quantization_config=quantization_config,
        device_map={"": LOCAL_RANK},
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    print(f"  Parameters: {model.num_parameters():,}", flush=True)

    # ── Resume adapter from Hub (cross-platform) ────────────────────────
    resume_checkpoint = None
    if args.resume_from_hub:
        # Adapter files are at the root of the Hub repo (no checkpoint subdirectories).
        # Strip any trailing /checkpoint-NNN suffix if present.
        repo_id = "/".join(args.resume_from_hub.split("/", 2)[:2])
        print(f"Loading pretrained adapter from Hub: {repo_id}", flush=True)
        model = PeftModel.from_pretrained(model, repo_id, is_trainable=True)
        print("  Adapter weights loaded. Optimizer state starts fresh.", flush=True)
    elif args.resume_from_hub_checkpoint:
        from huggingface_hub import HfApi, hf_hub_download

        # Accept "namespace/repo" or "namespace/repo/checkpoint-NNN"
        parts = args.resume_from_hub_checkpoint.split("/", 2)
        repo_id = f"{parts[0]}/{parts[1]}"
        explicit_subfolder = parts[2] if len(parts) == 3 else None

        api = HfApi(token=args.hub_token)
        all_files = api.list_repo_files(repo_id)

        # Find the checkpoint subfolder to resume from
        if explicit_subfolder:
            subfolder = explicit_subfolder
        else:
            # Collect all top-level directories from the repo file listing
            all_dirs = set(f.split("/")[0] for f in all_files if "/" in f)
            checkpoint_dirs = sorted(
                [d for d in all_dirs if re.match(r"^checkpoint-\d+$", d)],
                key=lambda x: int(x.split("-")[1]),
            )
            if checkpoint_dirs:
                subfolder = checkpoint_dirs[-1]
            elif "last-checkpoint" in all_dirs:
                subfolder = "last-checkpoint"
            else:
                raise FileNotFoundError(
                    f"No checkpoint directories found in {repo_id}. "
                    "Train with hub_strategy='checkpoint' first."
                )

        # Download every file — local_dir is the output root so the
        # subfolder prefix is preserved (e.g. last-checkpoint/optimizer.pt).
        os.makedirs(args.output_dir, exist_ok=True)
        for f in all_files:
            if f.startswith(subfolder + "/"):
                hf_hub_download(
                    repo_id=repo_id, filename=f,
                    local_dir=args.output_dir, token=args.hub_token,
                )

        resume_checkpoint = os.path.join(args.output_dir, subfolder)
        print(f"Resuming full state from {repo_id}/{subfolder} → {resume_checkpoint}", flush=True)
    elif args.resume_from:
        resume_checkpoint = args.resume_from
        print(f"Resuming from explicit checkpoint: {resume_checkpoint}", flush=True)
    elif args.resume:
        checkpoints = sorted(glob(os.path.join(args.output_dir, "checkpoint-*")),
                             key=lambda x: int(re.search(r"checkpoint-(\d+)", x).group(1)))
        if checkpoints:
            resume_checkpoint = checkpoints[-1]
            print(f"Resuming from latest checkpoint: {resume_checkpoint}", flush=True)
        else:
            print("No checkpoints found, starting fresh.", flush=True)

    # ── LoRA config (only if not already a PeftModel from Hub) ──────────
    if not args.resume_from_hub:
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=[
                "in_proj", "out_proj",
                "q_proj", "k_proj", "v_proj", "o_proj",
                "w1", "w2", "w3",
            ],
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print(
        f"LoRA: r={args.lora_r}, alpha={args.lora_alpha}, "
        f"modules={['in_proj', 'out_proj', 'q/k/v/o_proj', 'w1/w2/w3']}",
        flush=True,
    )

    # ── Reporting (W&B) — rank 0 only in DDP ─────────────────────────────
    if args.wandb_project and LOCAL_RANK == 0:
        try:
            import wandb
            wandb.init(project=args.wandb_project)
            report_to = ["wandb"]
        except Exception as e:
            print(f"W&B skipped: {e}", flush=True)
            report_to = ["none"]
    else:
        report_to = ["none"]

    # ── Training args (no grad clip — avoids GradScaler bf16 kernel bug) ─
    push_hub = args.hub_output and not args.no_hub
    training_args = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
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
        optim=args.optim,
        fp16=False,
        bf16=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to=report_to,
        push_to_hub=push_hub,
        hub_model_id=args.hub_output,
        hub_token=args.hub_token,
        hub_strategy="checkpoint",
        save_only_model=False,
        load_best_model_at_end=bool(eval_dataset),
        metric_for_best_model="eval_loss" if eval_dataset else None,
        greater_is_better=False if eval_dataset else None,
        completion_only_loss=True,
        dataset_num_proc=args.dataset_num_proc,
        eos_token=tokenizer.eos_token,
        max_length=args.max_seq_length,
        packing=args.packing,
        eval_packing=False,
        ddp_backend="nccl",
        ddp_find_unused_parameters=False,
        dataloader_num_workers=2,
        seed=args.seed,
        data_seed=args.seed,
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
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

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
