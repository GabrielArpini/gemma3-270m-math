from typing import Dict, List, Optional, Tuple
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer

class MathDatasets:
    def __init__(
        self,
        dataset_names: List[str],
        sample_sizes: Optional[List[Optional[int]]] = None,
        column_patterns: Optional[List[Tuple[str, str]]] = None,
        seed: int = 42
    ):
        self.dataset_names = dataset_names
        self.sample_sizes = sample_sizes or [None] * len(dataset_names)  # Match length
        self.column_patterns = column_patterns or [
            ("problem", "solution"),
            ("problem", "generated_solution"),
            ("question", "answer"),
            ("prompt", "completion"),
        ]
        self.seed = seed
        self.datasets: Dict[str, any] = {}  
        self.skipped_empty = 0
        self.tokenizer = AutoTokenizer.from_pretrained("unsloth/gemma-3-270m-it")

    def load_normalize(self):
        """
        Load, normalize, optionally subsample, and store each dataset.
        """
        for name, sample_size in zip(self.dataset_names, self.sample_sizes):
            print(f"Loading and normalizing: {name}")
            if name == "openai/gsm8k":
                ds = load_dataset("openai/gsm8k", "main")
            else:
                ds = load_dataset(name, split="train")

            ds = ds.map(self._normalize_sample, num_proc=4)

            ds = ds.filter(lambda x: x is not None and x.get("text") is not None)

            if sample_size is not None:
                ds = ds.shuffle(seed=self.seed).select(range(min(sample_size, len(ds))))

            self.datasets[name] = ds

        return self.datasets

    def _normalize_sample(self, example):
        """
        Normalize sample by checking column patterns.
        Iterates through self.column_patterns to find matching columns.
        Returns {'text': formatted_chat_text} or None if invalid
        """
        problem = None
        cot = None

        for problem_col, solution_col in self.column_patterns:
            if problem_col in example and solution_col in example:
                problem = example[problem_col]
                cot = example[solution_col]
                break

        if problem is None or cot is None:
            print(f"Skipped sample. No matching columns. Available: {list(example.keys())}")
            return None

        problem = str(problem).strip()
        cot = str(cot).strip()

        if not problem or not cot:
            self.skipped_empty += 1
            return None

        messages = [
            {
                "role": "user",
                "content": f"{problem}\n\nSolve step by step and put your final answer within \\boxed{{}}."
            },
            {
                "role": "assistant",
                "content": cot
            }
        ]

        formatted_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

        return {"text": formatted_text}

    def mix_dataset(self):
        """
        Concatenate all processed datasets and shuffle.
        Returns a single unified Dataset.
        """
        if not self.datasets:
            raise ValueError("No datasets loaded. Call load_normalize() first.")
        
        all_ds = list(self.datasets.values())
        if not all_ds:
            raise ValueError("No valid datasets after normalization.")
        
        combined = concatenate_datasets(all_ds)
        return combined.shuffle(seed=self.seed)


if __name__ == '__main__':
    dataset_names = [
        "AI-MO/NuminaMath-CoT",
        "microsoft/orca-math-word-problems-200k",
        "omniomni/omni-math",
        "openai/gsm8k"
    ]
    sample_sizes = [200000, None, 300000]  # Cap large ones

    column_patterns = [
        ("problem", "solution"),
        ("problem", "generated_solution"),  # For OpenMathReasoning if used 
        ("question", "answer"),
        ("prompt", "completion"),
    ]
    math_ds = MathDatasets(dataset_names, sample_sizes, column_patterns=column_patterns)

    #math_ds = MathDatasets(dataset_names, sample_sizes)
    math_ds.load_normalize()
    if math_ds.skipped_empty > 0:
        print(f"Skipped {math_ds.skipped_empty} empty samples.")
    mixed_train = math_ds.mix_dataset()

    print(f"Mixed dataset ready! Size: {len(mixed_train)} examples")    


