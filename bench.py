"""Benchmark model using parlor puzzles."""

import argparse
import asyncio
import collections
import datetime
import enum
import json
import logging
import os
import re
import subprocess
import tempfile
import textwrap
import time
from dataclasses import dataclass
from decimal import Decimal

import aiohttp
import yaml

LOGGER = logging.getLogger(__name__)


class PuzzleStatus(enum.Enum):
    CORRECT = "correct"
    INCORRECT = "incorrect"
    ERROR = "error"
    MALFORMED = "malformed"


@dataclass
class Puzzle:
    """A parlor puzzle."""

    puzzle_id: int
    blue_statement: str
    white_statement: str
    black_statement: str
    solution: str

    @classmethod
    def load_all(cls):
        with open("puzzles.json") as f:
            return [
                cls(
                    puzzle_id=puzzle["id"],
                    blue_statement=puzzle["blue"],
                    white_statement=puzzle["white"],
                    black_statement=puzzle["black"],
                    solution=puzzle["solution"],
                )
                for puzzle in json.load(f)
            ]

    @property
    def prompt(self):
        return textwrap.dedent(f"""
            Let's play a logic puzzle.

            The rules are:

            * You are in the parlor.
            * The room contains a blue box, a white box, and a black box.
            * The blue box is on the left, the white box is in the middle, and the black box is on the right.
            * Each box displays a statement on its lid.
            * There will always be at least one box which displays a true statement.
            * There will always be at least one box which displays a false statement.
            * Exactly one box contains gems; the other 2 are empty.
            * The room also contains a wind-up key.

            The three boxes display these statements:

            * blue box: {self.blue_statement}
            * white box: {self.white_statement}
            * black box: {self.black_statement}

            Which box contains the gems?

            Think step-by-step before writing your answer in exactly this format: `<solution>white</solution>`.
            """).strip()

    def check_solution(self, response):
        matches = re.findall(
            r"<solution>([A-Za-z ]+)</solution>", response, re.IGNORECASE
        )
        try:
            solution = matches[-1].strip()
        except IndexError:
            return PuzzleStatus.MALFORMED
        if solution.lower() != self.solution.lower():
            return PuzzleStatus.INCORRECT
        return PuzzleStatus.CORRECT


class CliError(Exception):
    pass


class NullResponseError(Exception):
    pass


class MaxRetriesExceededError(Exception):
    pass


class Model:
    """Generic OpenAI-compatible model."""

    MAX_RETRIES = 6
    INITIAL_DELAY_SECONDS = 1

    def __init__(
        self,
        api_base,
        parameters=None,
        headers=None,
        prompt_token_price=0,
        completion_token_price=0,
    ):
        self._api_base = api_base
        self._parameters = parameters
        self._headers = headers
        self._prompt_token_price = prompt_token_price
        self._completion_token_price = completion_token_price
        self.prompt_tokens_used = 0
        self.completion_tokens_used = 0

    @classmethod
    def load_model(cls, models_file, model_name):
        config = yaml.safe_load(models_file)
        providers = config.get("providers", {})
        models_config = config.get("models", {})
        try:
            model_config = models_config[model_name]
        except KeyError:
            raise CliError(f"Model {model_name!r} not found in models file")
        provider_name = model_config["provider"]
        try:
            provider = providers[provider_name]
        except KeyError:
            raise CliError(f"Provider {provider_name} not found in models file")
        return cls(
            expand_vars(provider["api_base"]),
            parameters=model_config.get("parameters", {}),
            headers={
                key: expand_vars(value)
                for key, value in provider.get("headers", {}).items()
            },
            prompt_token_price=Decimal(model_config.get("prompt_token_price", "0")),
            completion_token_price=Decimal(
                model_config.get("completion_token_price", "0")
            ),
        )

    @property
    def cost(self):
        return (self._prompt_token_price * self.prompt_tokens_used / 1_000_000) + (
            self._completion_token_price * self.completion_tokens_used / 1_000_000
        )

    async def prompt(self, prompt, timeout_seconds=300):
        body = {
            "messages": [{"role": "user", "content": prompt}],
            **self._parameters,
        }

        # Use "total" timeout rather than sock_read since we are not using
        # streaming
        timeout = aiohttp.ClientTimeout(total=timeout_seconds)

        for attempt in range(self.MAX_RETRIES + 1):
            try:
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(
                        url=f"{self._api_base}/chat/completions",
                        headers=self._headers,
                        json=body,
                    ) as response:
                        response.raise_for_status()
                        response_json = await response.json()
                        if response_json is None:
                            raise NullResponseError
            except NullResponseError:
                LOGGER.warning("Received null response")
            except TimeoutError:
                LOGGER.warning("Request timed out")
            except (aiohttp.ClientConnectionError, aiohttp.ClientPayloadError) as e:
                LOGGER.warning("Client connection error: %s", e)
            except aiohttp.ClientResponseError as e:
                if e.status == 429 or e.status >= 500:
                    LOGGER.warning("Client response error: %s", e)
                else:
                    raise  # Don't retry 4xx errors
            else:
                break
            delay = self.INITIAL_DELAY_SECONDS * (2**attempt)
            LOGGER.warning("Backing off for %s seconds", delay)
            await asyncio.sleep(delay)
        else:
            raise MaxRetriesExceededError("Max retries exceeded")

        self.completion_tokens_used += response_json["usage"]["completion_tokens"]
        self.prompt_tokens_used += response_json["usage"]["prompt_tokens"]
        finish_reason = response_json["choices"][0]["finish_reason"]
        if finish_reason != "stop":
            LOGGER.warning("Unexpected finish reason: %s", finish_reason)
        return response_json["choices"][0]["message"]["content"]


def expand_vars(value):
    """Expand environment variable references in the input string.

    Supports:
    * `${VAR_NAME}`
    * `$VAR_NAME`
    """
    pattern = re.compile(r"\$(\w+)|\$\{([^}]+)\}")

    def replacer(match):
        var_name = match.group(1) or match.group(2)
        return os.environ.get(var_name)

    return pattern.sub(replacer, value)


def git_describe():
    try:
        return (
            subprocess.run(
                ["git", "describe", "--always", "--dirty"],
                check=True,
                capture_output=True,
            )
            .stdout.decode()
            .strip()
        )
    except Exception as e:
        LOGGER.exception("Error occurred getting git describe: %s", e)
        return None


def save_result(result, results_filename):
    try:
        with open(results_filename, "r") as f:
            results = yaml.safe_load(f)
    except FileNotFoundError:
        results = []

    results.append(result)

    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, dir=os.path.dirname(results_filename)
    ) as f:
        yaml.dump(results, f)
        temp_name = f.name

    os.replace(temp_name, results_filename)


def log_prompt_and_response(filename, model, puzzle, status, prompt, response):
    with open(filename, "a") as f:
        f.write(f"# Puzzle {puzzle.puzzle_id}\n\n")
        f.write(f"**Model:** `{model}`\n\n")
        f.write(f"**Status:** {status.value}\n\n")
        f.write("## Prompt\n\n")
        f.write(f"{prompt}\n\n")
        f.write("## Response\n\n")
        f.write(f"{response}\n\n")


async def main():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "model",
        help="Model to use",
    )
    parser.add_argument(
        "--models-file",
        type=argparse.FileType("r"),
        default="models.yaml",
        help="Path to models configuration file",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Number of concurrent requests",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of puzzles to solve (first N)",
    )
    parser.add_argument(
        "--puzzle-id",
        type=int,
        default=None,
        help="Solve a specific puzzle by its ID",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=3,
        help="Number of times to repeat each puzzle",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout in seconds for each request",
    )
    parser.add_argument(
        "--responses-file",
        type=str,
        default="responses.md",
        help="Log prompts and responses to a markdown file",
    )
    parser.add_argument(
        "--results-file",
        type=str,
        default="results.yaml",
        help="Path to the results file",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Disable saving results",
    )
    args = parser.parse_args()

    try:
        model = Model.load_model(args.models_file, args.model)
    except CliError as e:
        exit(f"Error: {e}")
    puzzles = Puzzle.load_all()
    if args.puzzle_id is not None:
        puzzles = [puzzle for puzzle in puzzles if puzzle.puzzle_id == args.puzzle_id]
    if args.limit is not None:
        puzzles = puzzles[: args.limit]
    puzzle_queue = collections.deque(
        [puzzle for puzzle in puzzles for _ in range(args.repeat)]
    )
    puzzle_statuses = []  # Store tuples of (puzzle_id, status)
    start_time = time.time()

    async def worker():
        while puzzle_queue:
            puzzle = puzzle_queue.popleft()

            LOGGER.info("Solving puzzle %s", puzzle.puzzle_id)
            try:
                response = await model.prompt(
                    puzzle.prompt,
                    timeout_seconds=args.timeout,
                )
            except Exception:
                LOGGER.exception("Error occurred solving puzzle %s", puzzle.puzzle_id)
                status = PuzzleStatus.ERROR
            else:
                status = puzzle.check_solution(response)
                LOGGER.info("Puzzle %s status: %s", puzzle.puzzle_id, status.value)
                log_prompt_and_response(
                    args.responses_file,
                    args.model,
                    puzzle,
                    status,
                    puzzle.prompt,
                    response,
                )

            puzzle_statuses.append((puzzle.puzzle_id, status))

    async with asyncio.TaskGroup() as tg:
        for _ in range(args.concurrency):
            tg.create_task(worker())

    elapsed = datetime.timedelta(seconds=round(time.time() - start_time))
    LOGGER.info("Finish time: %s", elapsed)

    result = {
        "model": args.model,
        "commit_hash": git_describe(),
        "date": str(datetime.date.today()),
        "cost_usd": float(model.cost),
        "prompt_tokens": model.prompt_tokens_used,
        "completion_tokens": model.completion_tokens_used,
        "puzzle_id": {
            status.value: sorted(
                puzzle_id
                for puzzle_id, puzzle_status in puzzle_statuses
                if puzzle_status == status
            )
            for status in PuzzleStatus
        },
    }
    print("---")  # visual separation from logs
    print(yaml.dump(result))
    if not args.no_save:
        save_result(result, args.results_file)


if __name__ == "__main__":
    asyncio.run(main())
