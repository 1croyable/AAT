from __future__ import annotations

r"""
Controlled generation seal test for the trained WikiText-103 AAT and Transformer.

The script does not contain either model implementation.  It restores each model
from the exact artifacts saved by its training run:

    runs/canonical_aat_wikitext_paper/
        best.pt
        source_snapshot.py
        model_config.json
        run_config.json

    runs/transformer_wikitext_paper/
        best.pt
        source_snapshot.py
        model_config.json
        run_config.json

Default run from the AttentionResearch directory:

    python generation_seal_test_wikitext103.py --device cuda --precision auto

The default protocol compares the two models on:

    * eight deterministic 32-token prompts from the WikiText-103 test split,
      each with an 80-token real continuation;
    * the three prompts used by the original training scripts;
    * greedy decoding;
    * top-k sampling with temperature=0.9, top-k=40, and three identical random
      number streams per prompt/model.

No repetition penalty, no-repeat-ngram rule, beam search, or architecture change
is applied.  The purpose is to expose rather than hide generation degeneration.

Outputs are written to runs/generation_seal_test/:

    generation_results.json
    generation_summary.csv
    generation_samples.csv
    generation_samples.txt
    generation_report.txt
    blind_review.csv
    blind_key.json
    test_config.json
    source_snapshot.py
"""

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import random
import shutil
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable, Sequence

import torch
import torch.nn.functional as F


TEST_VERSION = "wikitext103-controlled-generation-seal-v1"
LEGACY_PROMPTS = (
    "The history of science",
    "In the city of London",
    "Researchers discovered that",
)


@dataclass(frozen=True)
class PromptCase:
    case_id: str
    source: str
    prompt_ids: tuple[int, ...]
    prompt_text: str
    reference_ids: tuple[int, ...] | None
    reference_text: str | None
    dataset_row: int | None


@dataclass(frozen=True)
class LoadedRun:
    name: str
    run_dir: Path
    implementation_version: str
    source_sha256: str
    checkpoint_sha256: str
    model_config: dict[str, Any]
    train_config: dict[str, Any]
    parameters: int


def require_huggingface() -> tuple[Any, Any]:
    try:
        from datasets import load_dataset
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Missing dependencies. Run: pip install torch datasets transformers"
        ) from exc
    return load_dataset, AutoTokenizer


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def choose_precision(
    requested: str,
    device: torch.device,
) -> tuple[torch.dtype, str]:
    if device.type != "cuda":
        return torch.float32, "fp32"
    if requested == "auto":
        requested = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    if requested == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("bf16 was requested but this GPU does not support it")
        return torch.bfloat16, "bf16"
    if requested == "fp16":
        return torch.float16, "fp16"
    if requested == "fp32":
        return torch.float32, "fp32"
    raise ValueError(f"unknown precision: {requested}")


def autocast_context(device: torch.device, dtype: torch.dtype):
    return torch.autocast(
        device_type=device.type,
        dtype=dtype,
        enabled=(device.type == "cuda" and dtype != torch.float32),
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return payload


def atomic_write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_write_json(payload: Any, path: Path) -> None:
    atomic_write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        path,
    )


def write_csv(rows: Sequence[dict[str, Any]], path: Path) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def load_source_module(path: Path, model_name: str) -> ModuleType:
    unique = hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()[:12]
    module_name = f"_generation_seal_{model_name}_{unique}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not create an import specification for {path}")
    module = importlib.util.module_from_spec(spec)
    # Dataclass construction consults sys.modules while the module is executing.
    sys.modules[module_name] = module
    previous_dont_write_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    finally:
        sys.dont_write_bytecode = previous_dont_write_bytecode
    return module


def torch_load_checkpoint(path: Path) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "map_location": "cpu",
        "weights_only": False,
    }
    try:
        payload = torch.load(path, mmap=True, **kwargs)
    except TypeError:
        try:
            payload = torch.load(path, **kwargs)
        except TypeError:
            payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"{path} does not contain a checkpoint dictionary")
    return payload


def required_run_files(run_dir: Path) -> dict[str, Path]:
    files = {
        "checkpoint": run_dir / "best.pt",
        "source": run_dir / "source_snapshot.py",
        "model_config": run_dir / "model_config.json",
        "run_config": run_dir / "run_config.json",
    }
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "run directory is incomplete; missing:\n  " + "\n  ".join(missing)
        )
    return files


def load_model_from_run(
    *,
    name: str,
    run_dir: Path,
    class_name: str,
    device: torch.device,
    allow_source_mismatch: bool,
) -> tuple[torch.nn.Module, LoadedRun]:
    files = required_run_files(run_dir)
    model_config = load_json(files["model_config"])
    train_config = load_json(files["run_config"])
    source_hash = sha256_file(files["source"])
    checkpoint_hash = sha256_file(files["checkpoint"])
    module = load_source_module(files["source"], name)

    if not hasattr(module, "ModelConfig") or not hasattr(module, class_name):
        raise AttributeError(
            f"{files['source']} does not expose ModelConfig and {class_name}"
        )
    implementation_version = str(
        getattr(module, "IMPLEMENTATION_VERSION", "unavailable")
    )
    config_object = module.ModelConfig(**model_config)
    model = getattr(module, class_name)(config_object)

    checkpoint = torch_load_checkpoint(files["checkpoint"])
    checkpoint_model_config = checkpoint.get("model_config")
    checkpoint_train_config = checkpoint.get("train_config")
    checkpoint_version = str(checkpoint.get("implementation_version", "unavailable"))
    saved_source_hash = str(checkpoint.get("source_sha256", "unavailable"))

    if checkpoint_model_config != model_config:
        raise RuntimeError(
            f"{name}: best.pt model_config differs from model_config.json"
        )
    if checkpoint_train_config != train_config:
        raise RuntimeError(
            f"{name}: best.pt train_config differs from run_config.json"
        )
    if checkpoint_version != implementation_version:
        raise RuntimeError(
            f"{name}: checkpoint implementation {checkpoint_version!r} does not "
            f"match source snapshot {implementation_version!r}"
        )
    if saved_source_hash != source_hash:
        message = (
            f"{name}: checkpoint source SHA-256 {saved_source_hash} does not match "
            f"source_snapshot.py {source_hash}"
        )
        if not allow_source_mismatch:
            raise RuntimeError(message + "; pass --allow-source-mismatch to override")
        print(f"WARNING: {message}", flush=True)

    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise KeyError(f"{name}: checkpoint has no model_state_dict")
    model.load_state_dict(state_dict, strict=True)
    del checkpoint, state_dict
    model.to(device)
    model.eval()
    parameters = sum(parameter.numel() for parameter in model.parameters())
    metadata = LoadedRun(
        name=name,
        run_dir=run_dir.resolve(),
        implementation_version=implementation_version,
        source_sha256=source_hash,
        checkpoint_sha256=checkpoint_hash,
        model_config=model_config,
        train_config=train_config,
        parameters=parameters,
    )
    return model, metadata


def validate_paired_runs(
    aat: LoadedRun,
    transformer: LoadedRun,
    tokenizer_size: int,
) -> int:
    if aat.train_config.get("tokenizer_name") != transformer.train_config.get(
        "tokenizer_name"
    ):
        raise RuntimeError("the two runs used different tokenizers")
    if aat.train_config.get("dataset_name") != transformer.train_config.get(
        "dataset_name"
    ):
        raise RuntimeError("the two runs used different datasets")
    if aat.train_config.get("dataset_config") != transformer.train_config.get(
        "dataset_config"
    ):
        raise RuntimeError("the two runs used different dataset configurations")
    for run in (aat, transformer):
        if int(run.model_config["vocab_size"]) != tokenizer_size:
            raise RuntimeError(
                f"{run.name}: checkpoint vocab size {run.model_config['vocab_size']} "
                f"does not match tokenizer size {tokenizer_size}"
            )
    contexts = {
        int(aat.train_config["sequence_length"]),
        int(transformer.train_config["sequence_length"]),
    }
    if len(contexts) != 1:
        raise RuntimeError("the two runs used different training context lengths")
    return contexts.pop()


def select_wikitext_prompts(
    *,
    raw_split: Any,
    tokenizer: Any,
    count: int,
    prompt_tokens: int,
    reference_tokens: int,
    seed: int,
) -> list[PromptCase]:
    required = prompt_tokens + reference_tokens
    candidates: list[tuple[int, tuple[int, ...]]] = []
    for row_index, text in enumerate(raw_split["text"]):
        if not isinstance(text, str):
            continue
        stripped = text.strip()
        if not stripped or (stripped.startswith("=") and stripped.endswith("=")):
            continue
        ids = tokenizer(
            stripped,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )["input_ids"]
        if len(ids) >= required:
            candidates.append((row_index, tuple(int(token) for token in ids)))
    if len(candidates) < count:
        raise RuntimeError(
            f"only {len(candidates)} WikiText rows contain at least {required} "
            f"tokens; requested {count}"
        )
    rng = random.Random(seed)
    rng.shuffle(candidates)
    selected: list[PromptCase] = []
    for index, (row_index, ids) in enumerate(candidates[:count], start=1):
        prompt_ids = ids[:prompt_tokens]
        reference_ids = ids[prompt_tokens:required]
        selected.append(
            PromptCase(
                case_id=f"W{index:02d}",
                source="wikitext",
                prompt_ids=prompt_ids,
                prompt_text=tokenizer.decode(
                    prompt_ids,
                    clean_up_tokenization_spaces=False,
                    skip_special_tokens=True,
                ),
                reference_ids=reference_ids,
                reference_text=tokenizer.decode(
                    reference_ids,
                    clean_up_tokenization_spaces=False,
                    skip_special_tokens=True,
                ),
                dataset_row=row_index,
            )
        )
    return selected


def build_legacy_prompts(tokenizer: Any) -> list[PromptCase]:
    cases: list[PromptCase] = []
    for index, text in enumerate(LEGACY_PROMPTS, start=1):
        ids = tokenizer(
            text,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )["input_ids"]
        cases.append(
            PromptCase(
                case_id=f"L{index:02d}",
                source="legacy",
                prompt_ids=tuple(int(token) for token in ids),
                prompt_text=text,
                reference_ids=None,
                reference_text=None,
                dataset_row=None,
            )
        )
    return cases


def ngrams(tokens: Sequence[int], width: int) -> list[tuple[int, ...]]:
    if width <= 0:
        raise ValueError("ngram width must be positive")
    return [
        tuple(tokens[start : start + width])
        for start in range(0, len(tokens) - width + 1)
    ]


def distinct_n(tokens: Sequence[int], width: int) -> float:
    grams = ngrams(tokens, width)
    return len(set(grams)) / len(grams) if grams else 1.0


def max_identical_run(tokens: Sequence[int]) -> int:
    if not tokens:
        return 0
    maximum = 1
    current = 1
    for previous, token in zip(tokens, tokens[1:]):
        if token == previous:
            current += 1
            maximum = max(maximum, current)
        else:
            current = 1
    return maximum


def max_consecutive_cycle(
    tokens: Sequence[int],
    max_width: int = 8,
) -> tuple[int, int]:
    best_repeats = 1 if tokens else 0
    best_width = 0
    token_count = len(tokens)
    for width in range(1, min(max_width, token_count // 2) + 1):
        for start in range(0, token_count - 2 * width + 1):
            block = tuple(tokens[start : start + width])
            repeats = 1
            cursor = start + width
            while (
                cursor + width <= token_count
                and tuple(tokens[cursor : cursor + width]) == block
            ):
                repeats += 1
                cursor += width
            if repeats > best_repeats or (
                repeats == best_repeats and repeats > 1 and width < best_width
            ):
                best_repeats = repeats
                best_width = width
    return best_repeats, best_width


def longest_repeated_ngram(tokens: Sequence[int], maximum: int = 16) -> int:
    for width in range(min(maximum, len(tokens) // 2), 1, -1):
        grams = ngrams(tokens, width)
        if len(set(grams)) < len(grams):
            return width
    return 0


def generation_token_metrics(
    generated_ids: Sequence[int],
    prompt_ids: Sequence[int],
) -> dict[str, Any]:
    token_count = len(generated_ids)
    metrics: dict[str, Any] = {
        "generated_tokens": token_count,
        "unique_token_ratio": (
            len(set(generated_ids)) / token_count if token_count else 0.0
        ),
        "adjacent_repeat_rate": (
            sum(left == right for left, right in zip(generated_ids, generated_ids[1:]))
            / max(1, token_count - 1)
        ),
        "max_identical_token_run": max_identical_run(generated_ids),
        "longest_repeated_ngram": longest_repeated_ngram(generated_ids),
    }
    for width in (2, 3, 4):
        distinct = distinct_n(generated_ids, width)
        metrics[f"distinct_{width}"] = distinct
        metrics[f"repeated_{width}gram_fraction"] = 1.0 - distinct
        gram_counts = Counter(ngrams(generated_ids, width))
        metrics[f"max_{width}gram_count"] = max(gram_counts.values(), default=0)
    cycle_repeats, cycle_width = max_consecutive_cycle(generated_ids)
    metrics["max_cycle_repeats"] = cycle_repeats
    metrics["max_cycle_width"] = cycle_width
    generated_fourgrams = ngrams(generated_ids, 4)
    prompt_fourgrams = set(ngrams(prompt_ids, 4))
    metrics["prompt_copy_4gram_rate"] = (
        sum(gram in prompt_fourgrams for gram in generated_fourgrams)
        / len(generated_fourgrams)
        if generated_fourgrams
        else 0.0
    )
    metrics["degenerate_repetition"] = bool(
        metrics["max_identical_token_run"] >= 4
        or metrics["max_cycle_repeats"] >= 4
        or metrics["repeated_4gram_fraction"] >= 0.20
    )
    return metrics


@torch.inference_mode()
def score_reference(
    *,
    model: torch.nn.Module,
    prompt_ids: Sequence[int],
    reference_ids: Sequence[int],
    device: torch.device,
    autocast_dtype: torch.dtype,
    max_context: int,
) -> dict[str, float]:
    combined = list(prompt_ids) + list(reference_ids)
    if len(combined) > max_context:
        raise ValueError(
            f"prompt + reference has {len(combined)} tokens, exceeding "
            f"the controlled context limit {max_context}"
        )
    sequence = torch.tensor([combined], device=device, dtype=torch.long)
    with autocast_context(device, autocast_dtype):
        logits = model(sequence[:, :-1])
    targets = sequence[:, 1:]
    start = len(prompt_ids) - 1
    continuation_logits = logits[:, start:].float()
    continuation_targets = targets[:, start:]
    losses = F.cross_entropy(
        continuation_logits.reshape(-1, continuation_logits.shape[-1]),
        continuation_targets.reshape(-1),
        reduction="none",
    )
    mean_nll = float(losses.mean().item())
    accuracy = float(
        (
            continuation_logits.argmax(dim=-1) == continuation_targets
        ).float().mean().item()
    )
    return {
        "reference_nll": mean_nll,
        "reference_perplexity": math.exp(min(20.0, mean_nll)),
        "reference_token_accuracy": accuracy,
        "reference_tokens": int(continuation_targets.numel()),
    }


@torch.inference_mode()
def generate_one(
    *,
    model: torch.nn.Module,
    prompt_ids: Sequence[int],
    eos_token_id: int,
    device: torch.device,
    autocast_dtype: torch.dtype,
    max_new_tokens: int,
    max_context: int,
    mode: str,
    temperature: float,
    top_k: int,
    seed: int | None,
) -> tuple[list[int], dict[str, Any]]:
    if mode not in {"greedy", "sampling"}:
        raise ValueError(f"unknown decoding mode: {mode}")
    ids = torch.tensor([list(prompt_ids)], device=device, dtype=torch.long)
    if ids.shape[1] == 0:
        ids = torch.tensor([[eos_token_id]], device=device, dtype=torch.long)
    uniforms: list[float] = []
    if mode == "sampling":
        if seed is None:
            raise ValueError("sampling requires a seed")
        cpu_generator = torch.Generator(device="cpu").manual_seed(seed)
        uniforms = torch.rand(max_new_tokens, generator=cpu_generator).tolist()

    generated: list[int] = []
    model_entropies: list[float] = []
    sampling_entropies: list[float] = []
    top1_probabilities: list[float] = []
    chosen_model_nlls: list[float] = []
    started = time.perf_counter()

    for step in range(max_new_tokens):
        context = ids[:, -max_context:]
        with autocast_context(device, autocast_dtype):
            logits = model(context)[:, -1].float()
        raw_log_probabilities = F.log_softmax(logits, dim=-1)
        raw_probabilities = raw_log_probabilities.exp()
        top1_probabilities.append(float(raw_probabilities.max().item()))
        model_entropies.append(
            float(
                -(raw_probabilities * raw_log_probabilities)
                .sum(dim=-1)
                .item()
            )
        )

        if mode == "greedy":
            next_token = logits.argmax(dim=-1, keepdim=True)
            sampling_entropies.append(model_entropies[-1])
        else:
            scaled_logits = logits / max(temperature, 1e-5)
            k = min(max(1, top_k), scaled_logits.shape[-1])
            values, indices = torch.topk(scaled_logits, k=k, dim=-1)
            probabilities = F.softmax(values, dim=-1)
            sampling_entropies.append(
                float(
                    -(probabilities * probabilities.clamp_min(1e-30).log())
                    .sum(dim=-1)
                    .item()
                )
            )
            cumulative = probabilities.cumsum(dim=-1)
            uniform = torch.tensor(
                [[uniforms[step]]],
                device=device,
                dtype=cumulative.dtype,
            )
            sampled_rank = torch.searchsorted(
                cumulative.contiguous(),
                uniform,
                right=False,
            ).clamp_max(k - 1)
            next_token = indices.gather(-1, sampled_rank)

        token = int(next_token.item())
        chosen_model_nlls.append(float(-raw_log_probabilities[0, token].item()))
        generated.append(token)
        ids = torch.cat((ids, next_token), dim=1)
        if token == eos_token_id:
            break

    seconds = time.perf_counter() - started
    return generated, {
        "ended_on_eos": bool(generated and generated[-1] == eos_token_id),
        "generation_seconds": seconds,
        "mean_model_entropy": sum(model_entropies) / max(1, len(model_entropies)),
        "mean_sampling_entropy": (
            sum(sampling_entropies) / max(1, len(sampling_entropies))
        ),
        "mean_top1_probability": (
            sum(top1_probabilities) / max(1, len(top1_probabilities))
        ),
        "mean_chosen_model_nll": (
            sum(chosen_model_nlls) / max(1, len(chosen_model_nlls))
        ),
        "chosen_model_perplexity": math.exp(
            min(
                20.0,
                sum(chosen_model_nlls) / max(1, len(chosen_model_nlls)),
            )
        ),
    }


def evaluate_model(
    *,
    model_name: str,
    model: torch.nn.Module,
    prompts: Sequence[PromptCase],
    tokenizer: Any,
    device: torch.device,
    autocast_dtype: torch.dtype,
    max_context: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    sampling_seeds: Sequence[int],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]]]:
    records: list[dict[str, Any]] = []
    reference_scores: dict[str, dict[str, float]] = {}
    for prompt in prompts:
        if prompt.reference_ids is not None:
            reference_scores[prompt.case_id] = score_reference(
                model=model,
                prompt_ids=prompt.prompt_ids,
                reference_ids=prompt.reference_ids,
                device=device,
                autocast_dtype=autocast_dtype,
                max_context=max_context,
            )

    decoding_jobs: list[tuple[str, str, int | None]] = [
        ("greedy", "greedy", None)
    ]
    decoding_jobs.extend(
        ("sampling", f"sampling_seed_{seed}", int(seed))
        for seed in sampling_seeds
    )
    total = len(prompts) * len(decoding_jobs)
    completed = 0
    for prompt_index, prompt in enumerate(prompts):
        for family, decoding, seed in decoding_jobs:
            effective_seed = (
                None
                if seed is None
                else int(seed) + prompt_index * 1_000_003
            )
            generated_ids, diagnostic_metrics = generate_one(
                model=model,
                prompt_ids=prompt.prompt_ids,
                eos_token_id=int(tokenizer.eos_token_id),
                device=device,
                autocast_dtype=autocast_dtype,
                max_new_tokens=max_new_tokens,
                max_context=max_context,
                mode=family,
                temperature=temperature,
                top_k=top_k,
                seed=effective_seed,
            )
            token_metrics = generation_token_metrics(
                generated_ids,
                prompt.prompt_ids,
            )
            continuation = tokenizer.decode(
                generated_ids,
                clean_up_tokenization_spaces=False,
                skip_special_tokens=True,
            )
            full_text = tokenizer.decode(
                list(prompt.prompt_ids) + generated_ids,
                clean_up_tokenization_spaces=False,
                skip_special_tokens=True,
            )
            record: dict[str, Any] = {
                "record_id": f"{prompt.case_id}__{decoding}",
                "model": model_name,
                "prompt_case_id": prompt.case_id,
                "prompt_source": prompt.source,
                "dataset_row": prompt.dataset_row,
                "decoding_family": family,
                "decoding": decoding,
                "sampling_seed": seed,
                "effective_sampling_seed": effective_seed,
                "temperature": temperature if family == "sampling" else None,
                "top_k": top_k if family == "sampling" else None,
                "prompt_tokens": len(prompt.prompt_ids),
                "prompt": prompt.prompt_text,
                "reference": prompt.reference_text,
                "continuation": continuation,
                "full_text": full_text,
                **token_metrics,
                **diagnostic_metrics,
            }
            if prompt.case_id in reference_scores:
                record.update(reference_scores[prompt.case_id])
            records.append(record)
            completed += 1
            print(
                f"[{model_name}] {completed:>3}/{total} "
                f"{prompt.case_id} {decoding}: "
                f"tokens={token_metrics['generated_tokens']} "
                f"distinct4={token_metrics['distinct_4']:.3f} "
                f"flag={token_metrics['degenerate_repetition']}",
                flush=True,
            )
    return records, reference_scores


def mean(values: Iterable[float]) -> float | None:
    materialized = list(values)
    return sum(materialized) / len(materialized) if materialized else None


def make_summary_rows(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    models = sorted({str(record["model"]) for record in records})
    sources = ("all", "wikitext", "legacy")
    exact_decodings = sorted({str(record["decoding"]) for record in records})
    decoding_groups = exact_decodings + ["sampling_all"]
    numeric_means = (
        "generated_tokens",
        "unique_token_ratio",
        "distinct_2",
        "distinct_3",
        "distinct_4",
        "repeated_4gram_fraction",
        "adjacent_repeat_rate",
        "max_identical_token_run",
        "max_cycle_repeats",
        "longest_repeated_ngram",
        "prompt_copy_4gram_rate",
        "mean_model_entropy",
        "mean_sampling_entropy",
        "mean_top1_probability",
        "mean_chosen_model_nll",
        "generation_seconds",
    )
    for model in models:
        for source in sources:
            for decoding_group in decoding_groups:
                subset = [
                    record
                    for record in records
                    if record["model"] == model
                    and (source == "all" or record["prompt_source"] == source)
                    and (
                        (
                            decoding_group == "sampling_all"
                            and record["decoding_family"] == "sampling"
                        )
                        or record["decoding"] == decoding_group
                    )
                ]
                if not subset:
                    continue
                row: dict[str, Any] = {
                    "model": model,
                    "prompt_source": source,
                    "decoding": decoding_group,
                    "samples": len(subset),
                    "degenerate_repetition_count": sum(
                        bool(record["degenerate_repetition"]) for record in subset
                    ),
                    "degenerate_repetition_rate": mean(
                        float(bool(record["degenerate_repetition"]))
                        for record in subset
                    ),
                    "eos_rate": mean(
                        float(bool(record["ended_on_eos"])) for record in subset
                    ),
                }
                for key in numeric_means:
                    value = mean(float(record[key]) for record in subset)
                    output_key = key if key.startswith("mean_") else f"mean_{key}"
                    row[output_key] = value

                # Reference scores are identical across decoding seeds.  Deduplicate
                # by prompt before aggregating their conditional NLL.
                unique_reference: dict[str, dict[str, Any]] = {}
                for record in subset:
                    if "reference_nll" in record:
                        unique_reference[str(record["prompt_case_id"])] = record
                reference_nll = mean(
                    float(record["reference_nll"])
                    for record in unique_reference.values()
                )
                row["reference_tokens"] = sum(
                    int(record["reference_tokens"])
                    for record in unique_reference.values()
                )
                row["reference_nll"] = reference_nll
                row["reference_perplexity"] = (
                    math.exp(min(20.0, reference_nll))
                    if reference_nll is not None
                    else None
                )
                row["reference_token_accuracy"] = mean(
                    float(record["reference_token_accuracy"])
                    for record in unique_reference.values()
                )
                rows.append(row)
    return rows


def find_summary(
    rows: Sequence[dict[str, Any]],
    *,
    model: str,
    source: str,
    decoding: str,
) -> dict[str, Any]:
    matches = [
        row
        for row in rows
        if row["model"] == model
        and row["prompt_source"] == source
        and row["decoding"] == decoding
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one summary for {model}/{source}/{decoding}, got {len(matches)}"
        )
    return matches[0]


def make_decision(summary_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    aat_sampling = find_summary(
        summary_rows,
        model="aat",
        source="wikitext",
        decoding="sampling_all",
    )
    transformer_sampling = find_summary(
        summary_rows,
        model="transformer",
        source="wikitext",
        decoding="sampling_all",
    )
    aat_count = int(aat_sampling["degenerate_repetition_count"])
    transformer_count = int(
        transformer_sampling["degenerate_repetition_count"]
    )
    rate_delta = float(aat_sampling["degenerate_repetition_rate"]) - float(
        transformer_sampling["degenerate_repetition_rate"]
    )
    count_delta = aat_count - transformer_count

    if aat_count >= 2 and count_delta >= 2 and rate_delta >= 0.10:
        status = "aat_repetition_disadvantage_observed"
        interpretation = (
            "AAT repetition degeneration was reproducibly worse on matched "
            "WikiText prompts and sampling streams. Inspect logits and memory-state "
            "dynamics next; this result alone still does not identify an architecture fix."
        )
    elif abs(count_delta) <= 1 and abs(rate_delta) < 0.10:
        status = "no_clear_aat_repetition_disadvantage"
        interpretation = (
            "The earlier repeated sample did not become a clear controlled AAT "
            "disadvantage. Keep the canonical architecture frozen."
        )
    else:
        status = "inconclusive"
        interpretation = (
            "The controlled result is mixed. Increase prompts or sampling seeds "
            "before changing the canonical architecture."
        )
    return {
        "primary_population": "wikitext prompts, all sampling seeds",
        "status": status,
        "interpretation": interpretation,
        "aat_flagged": aat_count,
        "transformer_flagged": transformer_count,
        "flagged_count_delta_aat_minus_transformer": count_delta,
        "flagged_rate_delta_aat_minus_transformer": rate_delta,
        "automatic_flag_definition": (
            "max identical-token run >= 4 OR max consecutive cycle repeats >= 4 "
            "OR repeated 4-gram fraction >= 0.20"
        ),
        "important_limit": (
            "Automated repetition metrics do not measure semantic coherence; "
            "use blind_review.csv for the paired human judgment."
        ),
    }


def make_blind_review(
    records: Sequence[dict[str, Any]],
    *,
    blind_seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    paired: dict[str, dict[str, dict[str, Any]]] = {}
    for record in records:
        paired.setdefault(str(record["record_id"]), {})[
            str(record["model"])
        ] = record

    review_rows: list[dict[str, Any]] = []
    key: dict[str, Any] = {}
    for pair_id in sorted(paired):
        models = paired[pair_id]
        if set(models) != {"aat", "transformer"}:
            raise RuntimeError(f"incomplete blind pair: {pair_id}")
        stable_seed = int.from_bytes(
            hashlib.sha256(f"{blind_seed}:{pair_id}".encode("utf-8")).digest()[:8],
            byteorder="big",
        )
        rng = random.Random(stable_seed)
        a_model, b_model = (
            ("aat", "transformer")
            if rng.random() < 0.5
            else ("transformer", "aat")
        )
        source = models["aat"]
        blind_id = f"B{len(review_rows) + 1:03d}"
        review_rows.append(
            {
                "blind_id": blind_id,
                "prompt_source": source["prompt_source"],
                "decoding": source["decoding"],
                "prompt": source["prompt"],
                "reference": source["reference"] or "",
                "sample_A": models[a_model]["continuation"],
                "sample_B": models[b_model]["continuation"],
                "fluency_winner_A_B_tie": "",
                "coherence_winner_A_B_tie": "",
                "repetition_winner_A_B_tie": "",
                "notes": "",
            }
        )
        key[blind_id] = {
            "pair_id": pair_id,
            "sample_A_model": a_model,
            "sample_B_model": b_model,
        }
    return review_rows, key


def samples_as_text(records: Sequence[dict[str, Any]]) -> str:
    by_pair: dict[str, dict[str, dict[str, Any]]] = {}
    for record in records:
        by_pair.setdefault(str(record["record_id"]), {})[
            str(record["model"])
        ] = record
    lines: list[str] = []
    separator = "=" * 110
    for pair_id in sorted(by_pair):
        pair = by_pair[pair_id]
        source = pair["aat"]
        lines.extend(
            [
                separator,
                (
                    f"{pair_id} | source={source['prompt_source']} | "
                    f"decoding={source['decoding']}"
                ),
                separator,
                f"PROMPT:\n{source['prompt']}",
            ]
        )
        if source.get("reference"):
            lines.append(f"\nREFERENCE:\n{source['reference']}")
        for model_name in ("aat", "transformer"):
            record = pair[model_name]
            lines.extend(
                [
                    (
                        f"\n[{model_name.upper()}] "
                        f"flag={record['degenerate_repetition']} "
                        f"distinct4={record['distinct_4']:.3f} "
                        f"repeat4={record['repeated_4gram_fraction']:.3f} "
                        f"max_run={record['max_identical_token_run']} "
                        f"max_cycle={record['max_cycle_repeats']}"
                    ),
                    record["continuation"],
                ]
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def report_as_text(
    *,
    metadata: Sequence[LoadedRun],
    summary_rows: Sequence[dict[str, Any]],
    decision: dict[str, Any],
    config: dict[str, Any],
) -> str:
    lines = [
        "=" * 110,
        "WIKITEXT-103 CONTROLLED GENERATION SEAL TEST",
        "=" * 110,
        f"test_version={TEST_VERSION}",
        (
            f"wiki_prompts={config['wiki_prompts']} "
            f"legacy_prompts={config['include_legacy_prompts']} "
            f"prompt_tokens={config['prompt_tokens']} "
            f"reference_tokens={config['reference_tokens']} "
            f"generate_tokens={config['generate_tokens']}"
        ),
        (
            f"sampling: temperature={config['temperature']} "
            f"top_k={config['top_k']} seeds={config['sampling_seeds']}"
        ),
        "No repetition penalty or no-repeat-ngram constraint was used.",
        "",
        "RESTORED RUNS",
    ]
    for run in metadata:
        lines.append(
            f"{run.name:>11}: parameters={run.parameters:,} "
            f"implementation={run.implementation_version} "
            f"source={run.source_sha256[:12]} checkpoint={run.checkpoint_sha256[:12]}"
        )
    lines.extend(
        [
            "",
            "PRIMARY AUTOMATED SUMMARY",
            (
                f"{'model':<12} {'source':<10} {'decoding':<14} "
                f"{'n':>4} {'flag':>8} {'distinct4':>10} {'repeat4':>9} "
                f"{'max_run':>8} {'max_cycle':>10} {'ref_ppl':>9}"
            ),
        ]
    )
    selected = [
        row
        for row in summary_rows
        if row["prompt_source"] in {"wikitext", "legacy"}
        and row["decoding"] in {"greedy", "sampling_all"}
    ]
    for row in selected:
        reference_ppl = row["reference_perplexity"]
        reference_text = (
            f"{reference_ppl:.2f}" if reference_ppl is not None else "-"
        )
        lines.append(
            f"{row['model']:<12} {row['prompt_source']:<10} "
            f"{row['decoding']:<14} {row['samples']:>4} "
            f"{row['degenerate_repetition_rate']:>7.1%} "
            f"{row['mean_distinct_4']:>10.3f} "
            f"{row['mean_repeated_4gram_fraction']:>9.3f} "
            f"{row['mean_max_identical_token_run']:>8.2f} "
            f"{row['mean_max_cycle_repeats']:>10.2f} "
            f"{reference_text:>9}"
        )
    lines.extend(
        [
            "",
            "DECISION",
            f"status={decision['status']}",
            decision["interpretation"],
            (
                f"AAT flagged={decision['aat_flagged']}; "
                f"Transformer flagged={decision['transformer_flagged']}; "
                f"rate delta={decision['flagged_rate_delta_aat_minus_transformer']:+.1%}"
            ),
            "",
            "Automatic severe-repetition flag:",
            decision["automatic_flag_definition"],
            "",
            "Limit:",
            decision["important_limit"],
            "",
            "Use generation_samples.txt for the complete paired text and "
            "blind_review.csv for a model-hidden human comparison.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a controlled generation seal test on trained canonical AAT "
            "and Transformer WikiText-103 checkpoints."
        )
    )
    parser.add_argument(
        "--aat-run-dir",
        type=Path,
        default=Path("runs/canonical_aat_wikitext_paper"),
    )
    parser.add_argument(
        "--transformer-run-dir",
        type=Path,
        default=Path("runs/transformer_wikitext_paper"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/generation_seal_test"),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--precision",
        choices=("auto", "fp32", "fp16", "bf16"),
        default="auto",
    )
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--dataset-split", default="test")
    parser.add_argument("--wiki-prompts", type=int, default=8)
    parser.add_argument("--prompt-tokens", type=int, default=32)
    parser.add_argument("--reference-tokens", type=int, default=80)
    parser.add_argument("--generate-tokens", type=int, default=80)
    parser.add_argument("--prompt-selection-seed", type=int, default=20260727)
    parser.add_argument(
        "--sampling-seeds",
        type=int,
        nargs="+",
        default=(1000, 1001, 1002),
    )
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--blind-seed", type=int, default=20260727)
    parser.add_argument("--no-legacy-prompts", action="store_true")
    parser.add_argument("--allow-source-mismatch", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    if min(
        args.wiki_prompts,
        args.prompt_tokens,
        args.reference_tokens,
        args.generate_tokens,
        args.top_k,
    ) <= 0:
        raise ValueError("prompt counts, token counts, and top-k must be positive")
    if args.temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if not args.sampling_seeds:
        raise ValueError("at least one sampling seed is required")
    if len(set(args.sampling_seeds)) != len(args.sampling_seeds):
        raise ValueError("sampling seeds must be unique")

    device = choose_device(args.device)
    autocast_dtype, precision_name = choose_precision(args.precision, device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    load_dataset, AutoTokenizer = require_huggingface()

    # Read the small run manifests before loading the much larger checkpoints.
    aat_files = required_run_files(args.aat_run_dir)
    transformer_files = required_run_files(args.transformer_run_dir)
    aat_run_config = load_json(aat_files["run_config"])
    transformer_run_config = load_json(transformer_files["run_config"])
    common_contexts = {
        int(aat_run_config["sequence_length"]),
        int(transformer_run_config["sequence_length"]),
    }
    if len(common_contexts) != 1:
        raise RuntimeError("the run manifests specify different context lengths")
    max_context = common_contexts.pop()
    if args.prompt_tokens + args.reference_tokens > max_context:
        raise ValueError(
            "prompt-tokens + reference-tokens must not exceed the common "
            f"training context ({max_context})"
        )
    tokenizer_names = {
        str(aat_run_config["tokenizer_name"]),
        str(transformer_run_config["tokenizer_name"]),
    }
    if len(tokenizer_names) != 1:
        raise RuntimeError("the run manifests specify different tokenizers")
    tokenizer_name = tokenizer_names.pop()
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        cache_dir=str(args.cache_dir) if args.cache_dir else None,
        use_fast=True,
    )
    if tokenizer.eos_token_id is None:
        raise RuntimeError("the tokenizer must define eos_token_id")

    dataset_names = {
        str(aat_run_config["dataset_name"]),
        str(transformer_run_config["dataset_name"]),
    }
    dataset_configs = {
        str(aat_run_config["dataset_config"]),
        str(transformer_run_config["dataset_config"]),
    }
    if len(dataset_names) != 1 or len(dataset_configs) != 1:
        raise RuntimeError("the run manifests specify different datasets")
    dataset_name = dataset_names.pop()
    dataset_config = dataset_configs.pop()
    print(
        f"Loading {dataset_name}/{dataset_config} split={args.dataset_split}...",
        flush=True,
    )
    raw_split = load_dataset(
        dataset_name,
        dataset_config,
        split=args.dataset_split,
        cache_dir=str(args.cache_dir) if args.cache_dir else None,
    )
    prompts = select_wikitext_prompts(
        raw_split=raw_split,
        tokenizer=tokenizer,
        count=args.wiki_prompts,
        prompt_tokens=args.prompt_tokens,
        reference_tokens=args.reference_tokens,
        seed=args.prompt_selection_seed,
    )
    if not args.no_legacy_prompts:
        prompts.extend(build_legacy_prompts(tokenizer))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy2(Path(__file__), args.output_dir / "source_snapshot.py")
    except OSError as exc:
        print(f"WARNING: could not save source snapshot: {exc}", flush=True)

    all_records: list[dict[str, Any]] = []
    all_reference_scores: dict[str, dict[str, dict[str, float]]] = {}
    run_metadata: list[LoadedRun] = []
    model_specs = (
        (
            "aat",
            args.aat_run_dir,
            "AATCausalLanguageModel",
        ),
        (
            "transformer",
            args.transformer_run_dir,
            "TransformerCausalLanguageModel",
        ),
    )
    for model_name, run_dir, class_name in model_specs:
        print(f"\nRestoring {model_name} from {run_dir}...", flush=True)
        model, metadata = load_model_from_run(
            name=model_name,
            run_dir=run_dir,
            class_name=class_name,
            device=device,
            allow_source_mismatch=args.allow_source_mismatch,
        )
        run_metadata.append(metadata)
        if len(run_metadata) == 2:
            validated_context = validate_paired_runs(
                run_metadata[0],
                run_metadata[1],
                len(tokenizer),
            )
            if validated_context != max_context:
                raise AssertionError("manifest context changed during restoration")
        print(
            f"{model_name}: implementation={metadata.implementation_version} "
            f"parameters={metadata.parameters:,} context={max_context}",
            flush=True,
        )
        records, reference_scores = evaluate_model(
            model_name=model_name,
            model=model,
            prompts=prompts,
            tokenizer=tokenizer,
            device=device,
            autocast_dtype=autocast_dtype,
            max_context=max_context,
            max_new_tokens=args.generate_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            sampling_seeds=args.sampling_seeds,
        )
        all_records.extend(records)
        all_reference_scores[model_name] = reference_scores
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary_rows = make_summary_rows(all_records)
    decision = make_decision(summary_rows)
    blind_rows, blind_key = make_blind_review(
        all_records,
        blind_seed=args.blind_seed,
    )
    config = {
        "test_version": TEST_VERSION,
        "device": str(device),
        "precision": precision_name,
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "dataset_name": dataset_name,
        "dataset_config": dataset_config,
        "dataset_split": args.dataset_split,
        "tokenizer": tokenizer_name,
        "common_context": max_context,
        "wiki_prompts": args.wiki_prompts,
        "include_legacy_prompts": not args.no_legacy_prompts,
        "prompt_tokens": args.prompt_tokens,
        "reference_tokens": args.reference_tokens,
        "generate_tokens": args.generate_tokens,
        "prompt_selection_seed": args.prompt_selection_seed,
        "sampling_seeds": list(args.sampling_seeds),
        "temperature": args.temperature,
        "top_k": args.top_k,
        "blind_seed": args.blind_seed,
        "repetition_penalty_used": False,
        "no_repeat_ngram_rule_used": False,
    }
    serialized_runs = [
        {
            **asdict(metadata),
            "run_dir": str(metadata.run_dir),
        }
        for metadata in run_metadata
    ]
    results = {
        "test_version": TEST_VERSION,
        "config": config,
        "runs": serialized_runs,
        "prompts": [asdict(prompt) for prompt in prompts],
        "reference_scores": all_reference_scores,
        "summary": summary_rows,
        "decision": decision,
        "samples": all_records,
    }
    output_dir = args.output_dir
    atomic_write_json(config, output_dir / "test_config.json")
    atomic_write_json(results, output_dir / "generation_results.json")
    atomic_write_json(blind_key, output_dir / "blind_key.json")
    write_csv(summary_rows, output_dir / "generation_summary.csv")
    write_csv(all_records, output_dir / "generation_samples.csv")
    write_csv(blind_rows, output_dir / "blind_review.csv")
    atomic_write_text(
        samples_as_text(all_records),
        output_dir / "generation_samples.txt",
    )
    atomic_write_text(
        report_as_text(
            metadata=run_metadata,
            summary_rows=summary_rows,
            decision=decision,
            config=config,
        ),
        output_dir / "generation_report.txt",
    )
    print("\n" + "=" * 110, flush=True)
    print("GENERATION SEAL TEST COMPLETE", flush=True)
    print("=" * 110, flush=True)
    print(f"status={decision['status']}", flush=True)
    print(decision["interpretation"], flush=True)
    print(f"Artifacts: {output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
