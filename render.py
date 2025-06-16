"""Render benchmark results to README.md."""

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import yaml
from adjustText import adjust_text


@dataclass
class ModelResult:
    model: str
    commit_hash: str
    date: str
    score_percent: float
    cost: float
    correct: int
    total: int


def process_results(results_path, strict=False):
    """Process results and return sorted model data."""
    results = yaml.safe_load(results_path.open())

    # Process and sort individual entries
    processed = []
    for entry in results:
        puzzle_id_data = entry["puzzle_id"]

        if strict:
            # Score: the number of puzzle IDs that were solved correctly in
            # every attempt.

            # Collect all unique puzzle IDs from all outcomes
            all_puzzle_ids = set()
            for outcome in puzzle_id_data.values():
                all_puzzle_ids.update(outcome)

            # For each puzzle, check if it was correct in every attempt
            strict_correct = 0
            for puzzle_id in all_puzzle_ids:
                total_attempts = 0
                correct_attempts = 0
                for outcome, ids in puzzle_id_data.items():
                    count_this = ids.count(puzzle_id)
                    total_attempts += count_this
                    if outcome == "correct":
                        correct_attempts = count_this

                if total_attempts > 0 and correct_attempts == total_attempts:
                    strict_correct += 1

            correct = strict_correct
            total = len(all_puzzle_ids) if all_puzzle_ids else 0
        else:
            # Score: the number of puzzles that were solved correctly
            correct = len(puzzle_id_data["correct"])
            total = len(
                puzzle_id_data["correct"]
                + puzzle_id_data["incorrect"]
                + puzzle_id_data["malformed"]
            )

        processed.append(
            ModelResult(
                model=entry["model"],
                commit_hash=entry["commit_hash"],
                date=entry["date"],
                score_percent=correct / total * 100,  # pass rate
                cost=entry["cost_usd"],
                correct=correct,
                total=total,
            )
        )

    # Sort by pass rate (desc), cost (asc), then date (desc)
    return sorted(
        processed, key=lambda x: (-x.score_percent, x.cost, -int(x.date.split("-")[2]))
    )


def generate_visualization(model_data):
    """Generate a Score vs Cost scatter plot and save as leaderboard.svg."""
    plt.figure(figsize=(10, 6))

    # Extract data
    models = []
    scores = []
    costs = []
    for result in model_data:
        models.append(
            result.model.split("/")[1] if "/" in result.model else result.model
        )
        scores.append(result.score_percent)  # percentage score
        costs.append(result.cost)  # cost USD

    # Create scatter plot with labels
    plt.scatter(
        costs,
        scores,
        alpha=0.6,
        edgecolors="w",
        linewidth=0.5,
        color="red",
    )

    # Create list of annotations for adjust_text
    texts = []
    for cost, score, model in zip(costs, scores, models):
        texts.append(
            plt.text(
                cost,
                score,
                model,
                fontsize=10,
                alpha=0.7,
                ha="center",
                va="center",
            )
        )

    # Adjust text labels to prevent overlaps
    adjust_text(
        texts,
        arrowprops=dict(
            arrowstyle="-",
            color="gray",
            alpha=0.3,
        ),
        expand=(1, 2),
    )

    # Style the plot
    plt.title("Blue Prince Parlor Puzzle - Score vs Cost", fontsize=14)
    plt.xlabel("Cost (USD)", fontsize=12)
    plt.ylabel("Score (%)", fontsize=12)

    # Format x-axis as decimal dollars
    plt.gca().xaxis.set_major_formatter("${x:1.2f}")

    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()

    # Save the visualization
    plt.savefig("leaderboard.svg")
    plt.close()


def update_readme(table_content, readme_path):
    """Update README with new results table."""
    content = readme_path.read_text()

    # Replace table between HTML comments
    new_content = re.sub(
        r"(<!-- BEGIN results -->\n).*?(\n<!-- END results -->)",
        f"\\1{table_content}\\2",
        content,
        flags=re.DOTALL,
    )

    readme_path.write_text(new_content)


RESULTS_PATH = Path("results.yaml")
README_PATH = Path("README.md")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Use strict scoring: puzzle must be correct every time to count.",
    )
    _args = parser.parse_args()

    model_data = process_results(RESULTS_PATH, strict=_args.strict)

    generate_visualization(model_data)

    # Generate markdown table
    table = [
        "| Rank | Model | Score | Cost (USD) |",
        "|------|-------|-------|------------|",
    ]

    for rank, result in enumerate(model_data, start=1):
        table.append(
            f"| {rank} | `{result.model}` | {result.score_percent:.0f}% ({result.correct}/{result.total}) | ${result.cost:.2f} |"
        )

    update_readme("\n".join(table), README_PATH)


if __name__ == "__main__":
    main()
