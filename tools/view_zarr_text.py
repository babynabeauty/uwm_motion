import argparse
import json
import os
import warnings
from dataclasses import asdict, dataclass
from typing import Iterable, Optional

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
warnings.filterwarnings("ignore", message="`clean_up_tokenization_spaces`.*")

import numpy as np
import zarr
from transformers import CLIPTokenizer


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_TOKENIZER_PATH = "/data/shared_workspace/LLM_weights/openai/clip-vit-base-patch32"
DEFAULT_TASK_LISTS = {
    "robocasa": os.path.join(REPO_ROOT, "configs/task_lists/robocasa_atomic_files.json"),
    "robotwin": os.path.join(REPO_ROOT, "configs/task_lists/robotwin_files.json"),
}


@dataclass
class ZarrTextSummary:
    task: str
    dataset: str
    zarr: str
    exists: bool
    has_language: bool
    num_episodes: int
    num_unique_texts: int
    unique_texts: list[str]
    samples: list[dict]
    error: str = ""


class ZarrTextViewer:
    def __init__(
        self,
        tokenizer_path: str = DEFAULT_TOKENIZER_PATH,
        skip_special_tokens: bool = True,
    ):
        self.tokenizer_path = tokenizer_path
        self.skip_special_tokens = skip_special_tokens
        self.tokenizer = CLIPTokenizer.from_pretrained(tokenizer_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @staticmethod
    def load_task_items(
        source: str = "robocasa",
        task_list_file: Optional[str] = None,
        enabled_only: bool = True,
        tasks: Optional[Iterable[str]] = None,
    ) -> list[dict]:
        if task_list_file is None:
            if source not in DEFAULT_TASK_LISTS:
                raise ValueError(f"Unknown source {source!r}; use one of {sorted(DEFAULT_TASK_LISTS)}")
            task_list_file = DEFAULT_TASK_LISTS[source]

        selected_tasks = set()
        for task in tasks or []:
            task = task.strip()
            if not task:
                continue
            selected_tasks.add(task)
            selected_tasks.add(task.replace(" ", "_"))
            selected_tasks.add(task.replace("_", " "))

        with open(task_list_file, "r") as f:
            data = json.load(f)

        items = []
        for item in data.get("items", []):
            if enabled_only and not item.get("enabled", True):
                continue
            task_name = str(item.get("task", "")).strip()
            dataset_name = str(item.get("dataset", "")).strip()
            zarr_path = str(item.get("zarr", "")).strip()
            if not zarr_path:
                continue
            if selected_tasks and task_name not in selected_tasks and dataset_name not in selected_tasks:
                continue
            items.append(
                {
                    "task": task_name,
                    "dataset": dataset_name or os.path.basename(zarr_path),
                    "zarr": zarr_path,
                    "enabled": bool(item.get("enabled", True)),
                }
            )
        return items

    def decode_input_ids(self, input_ids) -> str:
        ids = np.asarray(input_ids, dtype=np.int64).reshape(-1).tolist()
        return self.tokenizer.decode(
            ids,
            skip_special_tokens=self.skip_special_tokens,
        ).strip()

    def read_episode_texts(self, zarr_path: str, max_episodes: Optional[int] = None) -> list[str]:
        root = zarr.open(zarr_path, mode="r")
        if "meta" not in root or "input_ids" not in root["meta"]:
            raise KeyError(f"Missing meta/input_ids in {zarr_path}")

        input_ids = root["meta/input_ids"]
        limit = len(input_ids) if max_episodes is None else min(len(input_ids), max_episodes)
        return [self.decode_input_ids(input_ids[i]) for i in range(limit)]

    def summarize_zarr(
        self,
        zarr_path: str,
        task: str = "",
        dataset: str = "",
        max_unique: int = 20,
        max_samples: int = 8,
    ) -> ZarrTextSummary:
        if not os.path.exists(zarr_path):
            return ZarrTextSummary(
                task=task,
                dataset=dataset,
                zarr=zarr_path,
                exists=False,
                has_language=False,
                num_episodes=0,
                num_unique_texts=0,
                unique_texts=[],
                samples=[],
                error="zarr path does not exist",
            )

        try:
            root = zarr.open(zarr_path, mode="r")
            num_episodes = int(len(root["meta/episode_ends"])) if "episode_ends" in root["meta"] else 0
            if "input_ids" not in root["meta"]:
                return ZarrTextSummary(
                    task=task,
                    dataset=dataset,
                    zarr=zarr_path,
                    exists=True,
                    has_language=False,
                    num_episodes=num_episodes,
                    num_unique_texts=0,
                    unique_texts=[],
                    samples=[],
                    error="missing meta/input_ids",
                )

            input_ids = root["meta/input_ids"]
            unique_texts = []
            seen = set()
            samples = []
            for episode_idx in range(len(input_ids)):
                text = self.decode_input_ids(input_ids[episode_idx])
                if text not in seen:
                    seen.add(text)
                    if len(unique_texts) < max_unique:
                        unique_texts.append(text)
                if len(samples) < max_samples:
                    samples.append({"episode": episode_idx, "text": text})

            return ZarrTextSummary(
                task=task,
                dataset=dataset,
                zarr=zarr_path,
                exists=True,
                has_language=True,
                num_episodes=len(input_ids),
                num_unique_texts=len(seen),
                unique_texts=unique_texts,
                samples=samples,
            )
        except Exception as exc:
            return ZarrTextSummary(
                task=task,
                dataset=dataset,
                zarr=zarr_path,
                exists=True,
                has_language=False,
                num_episodes=0,
                num_unique_texts=0,
                unique_texts=[],
                samples=[],
                error=str(exc),
            )

    def summarize_items(
        self,
        items: Iterable[dict],
        max_unique: int = 20,
        max_samples: int = 8,
    ) -> list[ZarrTextSummary]:
        summaries = []
        for item in items:
            summaries.append(
                self.summarize_zarr(
                    item["zarr"],
                    task=item.get("task", ""),
                    dataset=item.get("dataset", ""),
                    max_unique=max_unique,
                    max_samples=max_samples,
                )
            )
        return summaries

    @staticmethod
    def print_summary(summary: ZarrTextSummary):
        status = "OK" if summary.has_language else "MISS"
        print(
            f"[{status}] task={summary.task or '<single-zarr>'} "
            f"episodes={summary.num_episodes} unique={summary.num_unique_texts}"
        )
        print(f"      zarr={summary.zarr}")
        if summary.error:
            print(f"      error={summary.error}")
            return
        if summary.unique_texts:
            print("      unique texts:")
            for text in summary.unique_texts:
                print(f"        - {text}")
        if summary.samples:
            print("      samples:")
            for sample in summary.samples:
                print(f"        ep{sample['episode']}: {sample['text']}")


def _parse_tasks(tasks_csv: str) -> list[str]:
    return [task.strip() for task in tasks_csv.split(",") if task.strip()]


def main():
    parser = argparse.ArgumentParser(
        description="Decode language tokens stored in zarr meta/input_ids."
    )
    parser.add_argument("--source", choices=sorted(DEFAULT_TASK_LISTS), default="robocasa")
    parser.add_argument("--task-list", default=None, help="Override task list json.")
    parser.add_argument("--zarr", default=None, help="View one zarr path directly.")
    parser.add_argument("--tasks", default="", help="Comma-separated task or dataset names.")
    parser.add_argument(
        "--include-disabled",
        action="store_true",
        help="Include disabled entries from task list.",
    )
    parser.add_argument("--max-unique", type=int, default=20)
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--tokenizer-path", default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args()

    viewer = ZarrTextViewer(tokenizer_path=args.tokenizer_path)

    if args.zarr:
        summaries = [
            viewer.summarize_zarr(
                args.zarr,
                max_unique=args.max_unique,
                max_samples=args.max_samples,
            )
        ]
    else:
        items = viewer.load_task_items(
            source=args.source,
            task_list_file=args.task_list,
            enabled_only=not args.include_disabled,
            tasks=_parse_tasks(args.tasks),
        )
        summaries = viewer.summarize_items(
            items,
            max_unique=args.max_unique,
            max_samples=args.max_samples,
        )

    if args.json:
        print(json.dumps([asdict(summary) for summary in summaries], indent=2, ensure_ascii=False))
    else:
        for idx, summary in enumerate(summaries):
            if idx:
                print()
            viewer.print_summary(summary)


if __name__ == "__main__":
    main()
