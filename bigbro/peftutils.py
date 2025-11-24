from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, TaskType, get_peft_model
from trl import SFTTrainer, SFTConfig
from datasets import Dataset
import wandb
import os


def fine_tune(
    model_name_or_path: str, peft_dataset: Dataset, wandb_key: str, hf_token: str
):
    wandb.login(key=wandb_key)

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "v_proj"],
    )

    batch_size = 1
    gradient_accumulation_steps = 16
    num_train_epochs = 1

    training_args = SFTConfig(
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size // 2 if batch_size // 2 != 0 else 1,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=5,
        num_train_epochs=num_train_epochs,
        learning_rate=2e-4,
        # fp16 = True,
        bf16=True,
        optim="adamw_torch",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=3152,
        output_dir="outputs",
        max_length=None,
        eval_strategy="steps",
        eval_steps=5,
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=1,
        report_to="wandb",
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, padding_side="left", token=hf_token
    )
    tokenizer.pad_token = tokenizer.eos_token

    lora_model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, dtype="auto", device_map="auto", token=hf_token
    )
    lora_model.generation_config.pad_token_id = tokenizer.pad_token_id

    lora_model = get_peft_model(lora_model, peft_config)
    lora_model.print_trainable_parameters()

    trainer = SFTTrainer(
        model=lora_model,
        train_dataset=peft_dataset["train"],
        eval_dataset=peft_dataset["test"],
        processing_class=tokenizer,
        args=training_args,
    )
    training_results = trainer.train()

    return training_results
