import argparse
from dotenv import load_dotenv
import os
from datautils import get_bigbro_prompts, get_peft_dataset
from llmutils import ChatBot
from evalutils import eval_queries_oai_async, calc_refusal
from peftutils import fine_tune
from raw_prompts import eval_prompt
from openai import AsyncOpenAI
from pathlib import Path
from sys import exit
import json
import time
import pandas as pd
import logging
from peft import AutoPeftModelForCausalLM
import torch

logger = logging.getLogger(__name__)


def timestamp() -> str:
    return time.strftime("%Y_%m_%d-%H_%M_%S")


def test_base_refusal(
    hf_token: str,
    test_model: str | Path,
    eval_model: str,
    eval_prompt: str,
):
    cwd = Path.cwd()
    dump_dir = cwd / "dumps"
    dump_dir.mkdir(exist_ok=True)
    dump_path = dump_dir / "demonstrations.csv"

    if not dump_path.is_file():
        logger.info("Unsafe demonstrations not found, creating...")
        profiling_prompts = get_bigbro_prompts(hf_token)

        UnsafeBot = ChatBot(test_model, hf_token)
        profiling_demonstrations = UnsafeBot(profiling_prompts)
        logger.info(f"Unsafe demonstrations created.")

        prompts_with_demos = pd.DataFrame(
            {"prompt": profiling_prompts, "response": profiling_demonstrations}
        )
        prompts_with_demos.to_csv(dump_path)
        logger.info(f"{timestamp()}: Dumped unsafe demonstrations.")

        client = AsyncOpenAI()
        openai_evaluations = eval_queries_oai_async(
            queries=profiling_prompts,
            responses=profiling_demonstrations,
            client=client,
            max_concurrent=10,
            evaluation_prompt=eval_prompt,
            evaluation_model=eval_model,
        )
        prompts_with_demos["evaluation"] = openai_evaluations
        prompts_with_demos.to_csv(dump_dir / "demonstrations_evaluated.csv")
    else:
        logger.info("Unsafe demonstrations found, loading...")
        prompts_with_demos = pd.read_csv(dump_path, index_col=0)
        logger.info(f"{timestamp()}: Loaded unsafe demonstrations.")

        client = AsyncOpenAI()
        openai_evaluations = eval_queries_oai_async(
            queries=prompts_with_demos["prompt"],
            responses=prompts_with_demos["response"],
            client=client,
            max_concurrent=10,
            evaluation_prompt=eval_prompt,
            evaluation_model=eval_model,
        )
        prompts_with_demos["evaluation"] = openai_evaluations
        prompts_with_demos.to_csv(dump_dir / "demonstrations_evaluated.csv")

    refusal_rate = calc_refusal(openai_evaluations)

    return refusal_rate


def test_ft_refusal(
    hf_token: str,
    test_model: AutoPeftModelForCausalLM,
    eval_model: str,
    eval_prompt: str,
):
    cwd = Path.cwd()
    dump_dir = cwd / "dumps"
    dump_dir.mkdir(exist_ok=True)
    dump_path = dump_dir / "ft_demonstrations.csv"

    logger.info("Creating demonstrations from the fine-tuned model...")
    profiling_prompts = get_peft_dataset(hf_token)
    profiling_prompts = [msgs[0:-1] for msgs in profiling_prompts["test"]["messages"]]

    SafeBot = ChatBot(test_model, hf_token)
    profiling_results = SafeBot(profiling_prompts)
    logger.info("Created.")

    prompts_with_demos = pd.DataFrame(
        {"prompt": profiling_prompts, "response": profiling_results}
    )
    prompts_with_demos.to_csv(dump_path)
    logger.info(f"{timestamp()}: Dumped the fine tuned model's responses.")

    client = AsyncOpenAI()
    openai_evaluations = eval_queries_oai_async(
        queries=profiling_prompts,
        responses=profiling_results,
        client=client,
        max_concurrent=10,
        evaluation_prompt=eval_prompt,
        evaluation_model=eval_model,
    )
    prompts_with_demos["evaluation"] = openai_evaluations
    prompts_with_demos.to_csv(dump_dir / "ft_demonstrations_evaluated.csv")

    refusal_rate = calc_refusal(openai_evaluations)

    return refusal_rate


def main():
    cwd = Path.cwd()
    logdir = cwd / "logs"
    logdir.mkdir(exist_ok=True)
    logging.basicConfig(
        filename=logdir / f"bigbro_{timestamp()}.log", level=logging.INFO
    )

    parser = argparse.ArgumentParser(
        description="Main script for running the building the dataset and running the experiments."
    )
    # parser.add_argument(
    #     "-d",
    #     "--debug",
    #     help="Debug mode with reduced dataset size.",
    #     action="store_true",
    # )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "-t",
        "--test",
        help="Test the language model with the BigBrother dataset.",
        action="store_true",
    )
    group.add_argument(
        "-f",
        "--fine_tune",
        help="Fine tune the language model with the BigBrother dataset.",
        action="store_true",
    )
    group.add_argument(
        "-s",
        "--safe_model_test",
        help="Test the fine tuned model with the held-out part of the BigBrother dataset.",
        action="store_true",
    )
    args = parser.parse_args()

    load_dotenv()
    hf_token = os.environ["HF_TOKEN"]
    local_model = os.environ["MODEL_NAME"]
    oai_model = os.environ["OPENAI_MODEL"]
    wandb_key = os.environ["WANDB_API_KEY"]

    if args.test:
        assert not args.fine_tune
        refusal_rate = test_base_refusal(hf_token, local_model, oai_model, eval_prompt)
        results_dir = cwd / "test_results"
        results_dir.mkdir(exist_ok=True)
        results_dict = {"refusal_rate": refusal_rate}
        results_file = results_dir / f"bigbro_results_{timestamp()}.json"
        with open(results_file, "w") as f:
            json.dump(results_dict, f)
        exit(0)
    elif args.fine_tune:
        assert not args.test
        dataset_with_safe_demonstrations = get_peft_dataset(hf_token)
        results = fine_tune(
            model_name_or_path=local_model,
            peft_dataset=dataset_with_safe_demonstrations,
            wandb_key=wandb_key,
            hf_token=hf_token,
        )

        ft_results_dir = cwd / "fine_tune_results"
        ft_results_dir.mkdir(exist_ok=True)
        ft_results_file = ft_results_dir / f"fine_tune_metrics_{timestamp()}.json"
        with open(ft_results_file, "w") as f:
            json.dump(results.metrics, f)
        exit(0)
    elif args.safe_model_test:
        assert not args.test and not args.fine_tune

        expected_checkpoint_dir = cwd / "outputs"
        if len(list(expected_checkpoint_dir.glob("*checkpoint*"))) != 1:
            raise FileNotFoundError(
                "The expected checkpoint file is not found, the file is either not unique or non-existent."
            )

        checkpoint = list(expected_checkpoint_dir.glob("*checkpoint*"))[0]
        local_model = AutoPeftModelForCausalLM.from_pretrained(
            checkpoint,
            dtype=torch.bfloat16,
            device_map="auto",
        )
        refusal_rate = test_ft_refusal(hf_token, local_model, oai_model, eval_prompt)
        results_dir = cwd / "ft_results"
        results_dir.mkdir(exist_ok=True)
        results_dict = {"refusal_rate": refusal_rate}
        results_file = results_dir / f"bigbro_ft_results_{timestamp()}.json"
        with open(results_file, "w") as f:
            json.dump(results_dict, f)
        exit(0)


if __name__ == "__main__":
    main()
