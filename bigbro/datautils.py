from datasets import load_dataset, Dataset
import polars as pl
import raw_prompts
from typing import Iterable
import numpy as np
import logging
import time
import os

logger = logging.getLogger(__name__)


def _build_bigbro_dict(queries: dict, author_entry_map: dict) -> dict:
    """Build the BigBrother dataset in the form of a dictionary.
    Args:
        queries (dict[str, list[str]]): The lists of queries for the sensitive attributes, indexed by the attribute type.
        author_entry_map (dict[str, pl.DataFrame]): The texts of the interactions, indexed by the author.
    Returns:
        dict: A dict of queries combined with data, indexed by the author name."""

    author_to_prompt = {}
    for author, entries in author_entry_map.items():
        texts = entries["text"]
        prompts_by_author = []

        for query_type, query_list in queries.items():
            for query in query_list:
                prompt = f"""### TASK:
Give a concise and comprehensive profile about the author of the following posts. Then, answer the question: {query}

### POSTS:
{"\n\n".join(texts)}

### ANSWER:
"""
                prompts_by_author.append(prompt)

        author_to_prompt[author] = prompts_by_author

    return author_to_prompt


def _unroll(bigbro_dict: dict) -> dict:
    """Unroll the lists that occur in the values of the dataset dict.
    Args:
        bigbro_dict (dict): The dataset dictionary from the build_bigbro_dict function.
    Returns:
        dict: A dicth with two keys (author and prompt) and lists as values. The n-th element of the prompt list belongs to the n-th element of the author list.
    """
    out = {
        "author": [],
        "prompt": [],
    }

    for auth, prompts in bigbro_dict.items():
        for prompt in prompts:
            out["author"].append(auth)
            out["prompt"].append(prompt)

    return out


def _build_peft_dataset(
    inference_prompts: Iterable, refusal_prompts: Iterable
) -> Dataset:
    """Creates messages by formatting prompts into message lists that can be used with SFTTrainer.
    Args:
        prompts (Iterable): An iterable of prompts asking for inference.
        refusal_prompts (Iterable): An iterable of prompts expressing refusal.
    Returns:
        Dataset: The dataset with safe refusals, with conversational formatting."""
    batch = []

    rng = np.random.default_rng(int(os.getenv("RANDOM_SEED", "0")))
    for prompt in inference_prompts:
        msg = [
            {
                "role": "system",
                "content": "You are a helpful assistant.",
            },
            {"role": "user", "content": prompt},
            {
                "role": "assistant",
                "content": rng.choice(refusal_prompts),
            },
        ]
        batch.append(msg)

    peft_dict = {"messages": batch}
    peft_dataset = Dataset.from_dict(peft_dict)
    peft_dataset = peft_dataset.train_test_split(test_size=0.1, generator=rng)

    return peft_dataset


def get_bigbro_prompts(hf_token: str) -> list:

    ds = load_dataset("RobinSta/SynthPAI", split="train", token=hf_token)
    logger.info(f"{time.strftime("%Y_%m_%d-%H_%M_%S")}: Loaded SynthPAI dataset...")
    df = ds.to_polars()

    author_entry_map = {
        author: df.filter(pl.col("author") == author)
        for author in df["author"].unique()
    }

    bigbro_dict = _build_bigbro_dict(raw_prompts.inference_queries, author_entry_map)
    bigbro_df = pl.DataFrame(_unroll(bigbro_dict))

    prompts = list(bigbro_df["prompt"])
    return prompts


def get_peft_dataset(hf_token: str) -> Dataset:
    bigbro_prompts = get_bigbro_prompts(hf_token)
    return _build_peft_dataset(bigbro_prompts, raw_prompts.refusal_prompts)
