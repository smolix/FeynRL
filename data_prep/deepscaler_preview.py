import argparse
import os
import re
import datasets


def create_prompt(question, system_prompt):
    """
    Create chat messages with optional system prompt.
    """
    if system_prompt:
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]
    return [{"role": "user", "content": question}]


def normalize_answer(answer_str):
    """
    Normalize answer text for exact-match reward comparison.
    """
    if answer_str is None:
        return None
    return str(answer_str).replace(",", "").replace("$", "").replace("\n", "").strip()


def extract_solution(solution_str):
    """
    Extract final numeric answer from "#### <answer>" if present.
    """
    if not solution_str:
        return None
    match = re.search(r"####\s*(-?[0-9.,]+)", str(solution_str))
    if match is None:
        return None
    return normalize_answer(match.group(1))


def make_training_answer(solution_trace, final_answer):
    """
    Build answer text used by training.
    """
    trace = str(solution_trace).strip() if solution_trace is not None else ""
    ans = str(final_answer).strip() if final_answer is not None else ""

    if trace and "####" in trace:
        return trace
    if trace and ans:
        return f"{trace}\n#### {ans}"
    if ans:
        return f"#### {ans}"
    if trace:
        return trace
    return ""


def make_map_fn(split, params):
    """
    Map examples into the framework format:
    prompt, answer, solution, split, index
    """

    def process_fn(example, idx):
        question = (
            example.get("problem")
            or example.get("question")
            or example.get("prompt")
            or example.get("input")
        )
        if question is None:
            raise ValueError(f"Missing question-like field. keys={list(example.keys())}")

        final_answer_raw = example.get("answer")
        solution_trace = example.get("solution")

        final_answer = normalize_answer(final_answer_raw)
        if final_answer is None:
            final_answer = extract_solution(solution_trace)

        answer_raw = make_training_answer(solution_trace, final_answer_raw)
        if not answer_raw:
            answer_raw = make_training_answer(solution_trace, final_answer)

        if final_answer is None:
            raise ValueError(f"Could not determine final answer for index={idx}")

        return {
            "prompt": create_prompt(str(question), params.system_prompt),
            "answer": answer_raw,
            "solution": final_answer,
            "split": split,
            "index": idx,
        }

    return process_fn


def create_file_name(params, split):
    fpart = "wsp" if params.system_prompt else "ns"
    return f"deepscaler_preview_processed_{params.run_id}_{fpart}_{split}.parquet"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_source", default="agentica-org/DeepScaleR-Preview-Dataset")
    parser.add_argument("--local_dir", required=True)
    parser.add_argument("--run_id", default="123245")
    parser.add_argument(
        "--system_prompt",
        default="You are a helpful assistant. Think step-by-step and output the final answer after '####'.",
    )
    parser.add_argument("--num_proc", type=int, default=4)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=123345)
    args = parser.parse_args()

    dataset = datasets.load_dataset(args.data_source)

    if "train" in dataset and "test" in dataset:
        train_val_split = dataset["train"].train_test_split(test_size=args.val_ratio, seed=args.seed)
        train_dataset = train_val_split["train"]
        val_dataset = train_val_split["test"]
        test_dataset = dataset["test"]
    elif "train" in dataset:
        full_dataset = dataset["train"]
        test_split = full_dataset.train_test_split(test_size=args.test_ratio, seed=args.seed)
        test_dataset = test_split["test"]
        remaining_dataset = test_split["train"]
        adjusted_val_ratio = args.val_ratio / (1.0 - args.test_ratio)
        val_split = remaining_dataset.train_test_split(test_size=adjusted_val_ratio, seed=args.seed)
        train_dataset = val_split["train"]
        val_dataset = val_split["test"]
    else:
        raise ValueError("Dataset must contain at least a 'train' split.")

    train_dataset = train_dataset.map(function=make_map_fn("train", params=args), with_indices=True, num_proc=args.num_proc)
    val_dataset = val_dataset.map(function=make_map_fn("val", params=args), with_indices=True, num_proc=args.num_proc)
    test_dataset = test_dataset.map(function=make_map_fn("test", params=args), with_indices=True, num_proc=args.num_proc)

    os.makedirs(args.local_dir, exist_ok=True)
    train_file_name = os.path.join(args.local_dir, create_file_name(args, "train"))
    val_file_name = os.path.join(args.local_dir, create_file_name(args, "val"))
    test_file_name = os.path.join(args.local_dir, create_file_name(args, "test"))
    train_dataset.to_parquet(train_file_name)
    val_dataset.to_parquet(val_file_name)
    test_dataset.to_parquet(test_file_name)

    # print samples in clean fashion
    print("\n\n")
    print(train_dataset[0])
    print(val_dataset[0])
    print(test_dataset[0])
    print("\n\n")

    print("")
    print(f"Train file: {train_file_name} with {len(train_dataset)} examples.")
    print(f"Val file: {val_file_name} with {len(val_dataset)} examples.")
    print(f"Test file: {test_file_name} with {len(test_dataset)} examples.")
    print("Done.")
