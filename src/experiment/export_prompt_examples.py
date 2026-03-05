import json
from typing import Any
from pathlib import Path

from make_prompts import MakePrompt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "experiment"


def collect_prompt_examples() -> list[dict[str, Any]]:
    """Gather examples for every axis/example_num combination."""
    axes = ["right-wing", "left-wing"]
    example_counts = [2, 4, 6]
    collected: list[dict[str, Any]] = []

    for eco in axes:
        for soc in axes:
            for count in example_counts:
                prompt_builder = MakePrompt(eco, soc, count)
                collected.append(
                    {
                        "eco_ideology": eco,
                        "soc_ideology": soc,
                        "example_num": count,
                        "examples": prompt_builder.examples,
                    }
                )

    return collected


def main() -> None:
    all_examples = collect_prompt_examples()
    output_path = DATA_DIR / "prompt_examples.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_examples, f, ensure_ascii=False, indent=2)
    print(f"Wrote examples to {output_path}")


if __name__ == "__main__":
    main()
