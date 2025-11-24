from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from collections.abc import Iterable
from peft import PeftModelForCausalLM
from tqdm import tqdm


class ChatBot:
    """Lightweight wrapper around HF Transformers."""

    def __init__(
        self, model: str | AutoModelForCausalLM | PeftModelForCausalLM, token: str
    ):
        """Initializes the ChatBot.
        Args:
            model (str | AutoModelForCausalLM | PeftModelForCausalLM): HF model name, or initialized model.
            token (str): HF token for authentication.
        """
        if isinstance(model, str):
            self.tokenizer = AutoTokenizer.from_pretrained(
                model, padding_side="left", token=token
            )
            self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = AutoModelForCausalLM.from_pretrained(
                model, dtype="auto", device_map="auto", token=token
            )
            self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
        elif isinstance(model, PeftModelForCausalLM):
            self.tokenizer = AutoTokenizer.from_pretrained(
                model.config.name_or_path, padding_side="left"
            )
            self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = model
            self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id

    def __call__(self, prompts):
        """Process the prompts and return the results.
        Args:
            prompts (list): List of prompts to be processed.
        Returns:
            list: List of results for each prompt."""
        batched_data = self._create_message_batch(prompts)
        results = self._process_batch(batched_data)
        return results

    def _create_message_batch(self, prompts: Iterable) -> list:
        """Creates messages by formatting prompts into message lists that can be passed to tokenizer.apply_chat_template().
        Args:
            prompts (Iterable): An iterable of prompts
        Returns:
            list: A list of messages."""
        batch = []

        for prompt in prompts:
            msg = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant.",
                },
                {"role": "user", "content": prompt},
            ]
            batch.append(msg)

        return batch

    def _process_batch(self, data: Iterable, batch_size: int = 8) -> list:
        """Process message data batch by batch.
        Args:
            data (Iterable): Iterable containing the messages.
            batch_size (int): Batch size.
        Returns:
            list: List of answers to the prompts."""
        self.model.eval()

        with torch.no_grad():
            answers = []
            for start_i in tqdm(range(0, len(data), batch_size)):
                model_inputs = self.tokenizer.apply_chat_template(
                    data[start_i : start_i + batch_size],
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                    padding=True,
                )
                model_inputs = {k: v.to("cuda") for k, v in model_inputs.items()}
                input_length = model_inputs["input_ids"].shape[1]
                generated_ids = self.model.generate(
                    model_inputs["input_ids"],
                    max_new_tokens=1024,
                    do_sample=False,
                    attention_mask=model_inputs["attention_mask"],
                )
                answers.extend(
                    self.tokenizer.batch_decode(
                        generated_ids[:, input_length:], skip_special_tokens=True
                    )
                )

            return answers
