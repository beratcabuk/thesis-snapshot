from tqdm import tqdm
import openai
import asyncio
from tenacity import retry
from tenacity.stop import stop_after_attempt
from tenacity.wait import wait_exponential_jitter
from tenacity.retry import retry_if_exception_type
from typing import Iterable
from llmutils import ChatBot
import os
import numpy as np
import logging
import time

logger = logging.getLogger(__name__)


def evaluate_responses(
    chatbot: ChatBot, responses: list[str], eval_prompt: str, prompts: list[str]
) -> float:
    """Evaluates the responses for refusal.
    Args:
        responses (list): An iterable containing the responses to evaluate.
        eval_prompt (str): Prompt to use for evaluation.
        prompts (list): Prompts that were asked for the responses.
    Returns:
        float: Refusal rate."""

    assert len(responses) == len(prompts), "responses, prompts should be of same len"

    N = len(responses)
    formatted_eval_prompts = [
        eval_prompt.format(QUESTION=prompts[i], RESPONSE=responses[i]) for i in range(N)
    ]

    evals = chatbot(formatted_eval_prompts)

    c = 0
    for e in evals:
        if "REFUSE" in e:
            c += 1
    return c / len(evals)


def eval_queries_oai(
    queries: Iterable[str],
    responses: Iterable[str],
    client: openai.OpenAI,
    prompt: str,
    model_name: str,
) -> list[str]:
    """Evaluates the queries using an OpenAAI model.
    Args:
        queries (Iterable[str]): Queries to evaluate.
        responses (Iterable[str]): Responses to evaluate.
        client (openai.OpenAI): The client for the OpenAI model.
        prompt (str): The evaluation prompt to be used.
    Returns:
        list: List of evaluations.
    """
    evals = []
    for i in tqdm(range(len(queries))):
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt.format(
                        QUESTION=queries[i], RESPONSE=responses[i]
                    ),
                }
            ],
            temperature=0,
        )
        evals.append(completion.choices[0].message.content)

    return evals


@retry(
    retry=retry_if_exception_type(openai.RateLimitError),
    wait=wait_exponential_jitter(max=32),
)
async def _async_process_one(
    semaphore: asyncio.Semaphore,
    client: openai.AsyncOpenAI,
    prompt: str,
    query: str,
    response: str,
    model_name: str,
) -> str:
    async with semaphore:
        completion = await client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt.format(QUESTION=query, RESPONSE=response),
                }
            ],
            temperature=0,
        )
    return completion.choices[0].message.content


@retry(stop=stop_after_attempt(10), wait=wait_exponential_jitter(max=32))
async def _mock_async_process_one(semaphore, query, *args):
    async with semaphore:
        await asyncio.sleep(2 * np.random.random())
        if np.random.random() <= 0.1:
            raise Exception("Mock exception occured.")
    return str(np.random.choice(["COMPLY", "REFUSE"]))


def eval_queries_oai_async(
    queries: Iterable[str],
    responses: Iterable[str],
    client: openai.AsyncOpenAI,
    max_concurrent: int,
    evaluation_prompt: str,
    evaluation_model: str,
) -> list[str]:
    if not isinstance(client, openai.AsyncOpenAI):
        raise TypeError(f"The client should be an AsyncOpenAI client.")

    sem = asyncio.Semaphore(max_concurrent)

    logger.info(f"{time.strftime("%Y_%m_%d-%H_%M_%S")}: Starting openai evals...")

    async def run_tasks():
        tasks = (
            _async_process_one(
                semaphore=sem,
                client=client,
                prompt=evaluation_prompt,
                query=query,
                response=response,
                model_name=evaluation_model,
            )
            for query, response in zip(queries, responses)
        )
        return await asyncio.gather(*tasks)

    return asyncio.run(run_tasks())


async def mock_eval_queries_oai_async(
    queries: Iterable[str],
    responses: Iterable[str],
    client: openai.AsyncOpenAI,
    max_concurrent: int,
    prompt: str,
    evaluation_model: str,
) -> list[str]:
    if not isinstance(client, openai.AsyncOpenAI):
        raise TypeError(f"The client should be an AsyncOpenAI client.")

    sem = asyncio.Semaphore(max_concurrent)

    return await asyncio.gather(
        *(
            _mock_async_process_one(
                sem, client, prompt, queries[i], responses[i], evaluation_model
            )
            for i in range(len(queries))
        )
    )


def calc_refusal(evals: list) -> float:
    c = 0
    for e in evals:
        if "REFUSE" in e:
            c += 1
    return c / len(evals)
