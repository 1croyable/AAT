from __future__ import annotations

"""Causal five-task comparison with the task query placed at sequence end."""

import copy
import math
import random
import time
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from aat.sequence import AAT, AATConfig

FIELD_COUNT = 3
PAD_TYPE = 0

ALL_TASKS = (
    "associative_retrieval",
    "latest_value_retrieval",
    "relative_order",
    "global_majority",
    "boolean_listops",
)

SEEDS = (0, 1, 2)
ENCODER_LAYERS = (1, 4)
DECODER_LAYERS = 4
D_MODEL = 96
HEADS = 8
RAYS = 8
AAT_FFN_DIM = 222
CHUNK_SIZE = 32
DROPOUT = 0.10
KAPPA = 6.0
SCORE_CLIP = 30.0
RECURRENT_LAYERS = 4
TRANSFORMER_LAYERS = 4


@dataclass(frozen=True)
class TrainConfig:
    train_count: int = 6_000
    val_count: int = 1_000
    test_count: int = 1_000
    long_count: int = 1_000
    stress_count: int = 1_000
    batch_size: int = 128
    epochs: int = 10
    learning_rate: float = 8e-4
    weight_decay: float = 1e-4
    warmup_fraction: float = 0.08
    grad_clip: float = 1.0

    @property
    def counts(self) -> tuple[int, int, int, int, int]:
        return (
            self.train_count,
            self.val_count,
            self.test_count,
            self.long_count,
            self.stress_count,
        )


@dataclass
class TaskBundle:
    name: str
    description: str
    field_sizes: tuple[int, int, int]
    num_classes: int
    chance_accuracy: float
    train: TensorDataset
    val: TensorDataset
    test: TensorDataset
    long: TensorDataset
    stress: TensorDataset
    split_descriptions: tuple[str, str, str]


@dataclass
class BenchmarkResult:
    task: str
    model: str
    seed: int
    parameters: int
    best_epoch: int
    best_val: float
    test: float
    long: float
    stress: float
    train_seconds: float


@dataclass(frozen=True)
class ModelSpec:
    name: str
    kind: str
    width: int
    layers: int
    parameters: int
    detail: str


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    center = mean(values)
    return math.sqrt(sum((value - center) ** 2 for value in values) / len(values))


def normalized_above_chance(accuracy: float, chance: float) -> float:
    return (accuracy - chance) / max(1.0 - chance, 1e-8)


def make_loader(
    dataset: TensorDataset,
    *,
    batch_size: int,
    shuffle: bool,
    device: torch.device,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )


def make_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    warmup_fraction: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup_steps = max(1, int(total_steps * warmup_fraction))

    def multiplier(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


AR_TYPE_QUERY = 1
AR_TYPE_RECORD = 2
AR_NUM_KEYS = 256
AR_NUM_VALUES = 4


def make_associative_retrieval_split(
    count: int,
    *,
    records: int,
    seed: int,
) -> TensorDataset:
    if records % AR_NUM_VALUES:
        raise ValueError("records must be divisible by AR_NUM_VALUES")
    if records >= AR_NUM_KEYS:
        raise ValueError("records must be less than AR_NUM_KEYS")

    rng = random.Random(seed)
    tokens = torch.zeros(count, 1 + records, FIELD_COUNT, dtype=torch.long)
    labels = torch.empty(count, dtype=torch.long)
    balanced_values = [
        value
        for value in range(AR_NUM_VALUES)
        for _ in range(records // AR_NUM_VALUES)
    ]

    for row in range(count):
        label = row % AR_NUM_VALUES
        keys = rng.sample(range(1, AR_NUM_KEYS + 1), records)
        target_slot = rng.randrange(records)
        query_key = keys[target_slot]
        values = balanced_values.copy()
        rng.shuffle(values)
        label_slot = values.index(label)
        values[target_slot], values[label_slot] = values[label_slot], values[target_slot]
        records_data = [
            (AR_TYPE_RECORD, keys[index], values[index] + 1)
            for index in range(records)
        ]
        rng.shuffle(records_data)
        tokens[row, 0] = torch.tensor((AR_TYPE_QUERY, query_key, 0))
        tokens[row, 1:] = torch.tensor(records_data)
        labels[row] = label

    order = torch.randperm(count, generator=torch.Generator().manual_seed(seed + 97))
    return TensorDataset(tokens.index_select(0, order), labels.index_select(0, order))


def build_associative_retrieval(
    counts: tuple[int, int, int, int, int],
    *,
    seed: int,
) -> TaskBundle:
    train, val, test, long, stress = counts
    return TaskBundle(
        name="associative_retrieval",
        description="Retrieve the value associated with a queried key.",
        field_sizes=(3, AR_NUM_KEYS + 1, AR_NUM_VALUES + 1),
        num_classes=AR_NUM_VALUES,
        chance_accuracy=1.0 / AR_NUM_VALUES,
        train=make_associative_retrieval_split(train, records=8, seed=seed + 101),
        val=make_associative_retrieval_split(val, records=8, seed=seed + 102),
        test=make_associative_retrieval_split(test, records=8, seed=seed + 103),
        long=make_associative_retrieval_split(long, records=16, seed=seed + 104),
        stress=make_associative_retrieval_split(stress, records=32, seed=seed + 105),
        split_descriptions=("8 records", "16 records", "32 records"),
    )


LVR_TYPE_QUERY = 1
LVR_TYPE_RECORD = 2
LVR_NUM_KEYS = 128
LVR_NUM_VALUES = 4


def make_latest_value_retrieval_split(
    count: int,
    *,
    records: int,
    query_occurrences: int,
    seed: int,
) -> TensorDataset:
    if query_occurrences < 2 or query_occurrences >= records:
        raise ValueError("query_occurrences must be in [2, records)")

    rng = random.Random(seed)
    tokens = torch.zeros(count, 1 + records, FIELD_COUNT, dtype=torch.long)
    labels = torch.empty(count, dtype=torch.long)

    for row in range(count):
        label = row % LVR_NUM_VALUES
        query_key = rng.randrange(1, LVR_NUM_KEYS + 1)
        occurrence_positions = sorted(rng.sample(range(records), query_occurrences))
        final_position = occurrence_positions[-1]
        records_data: list[tuple[int, int, int]] = []

        for position in range(records):
            if position in occurrence_positions:
                if position == final_position:
                    value = label
                else:
                    value = rng.choice(
                        [candidate for candidate in range(LVR_NUM_VALUES) if candidate != label]
                    )
                records_data.append((LVR_TYPE_RECORD, query_key, value + 1))
            else:
                key = rng.randrange(1, LVR_NUM_KEYS + 1)
                while key == query_key:
                    key = rng.randrange(1, LVR_NUM_KEYS + 1)
                records_data.append(
                    (LVR_TYPE_RECORD, key, rng.randrange(LVR_NUM_VALUES) + 1)
                )

        tokens[row, 0] = torch.tensor((LVR_TYPE_QUERY, query_key, 0))
        tokens[row, 1:] = torch.tensor(records_data)
        labels[row] = label

    order = torch.randperm(count, generator=torch.Generator().manual_seed(seed + 97))
    return TensorDataset(tokens.index_select(0, order), labels.index_select(0, order))


def build_latest_value_retrieval(
    counts: tuple[int, int, int, int, int],
    *,
    seed: int,
) -> TaskBundle:
    train, val, test, long, stress = counts
    return TaskBundle(
        name="latest_value_retrieval",
        description="Retrieve the value from the last occurrence of a repeated key.",
        field_sizes=(3, LVR_NUM_KEYS + 1, LVR_NUM_VALUES + 1),
        num_classes=LVR_NUM_VALUES,
        chance_accuracy=1.0 / LVR_NUM_VALUES,
        train=make_latest_value_retrieval_split(
            train, records=12, query_occurrences=3, seed=seed + 151
        ),
        val=make_latest_value_retrieval_split(
            val, records=12, query_occurrences=3, seed=seed + 152
        ),
        test=make_latest_value_retrieval_split(
            test, records=12, query_occurrences=3, seed=seed + 153
        ),
        long=make_latest_value_retrieval_split(
            long, records=24, query_occurrences=5, seed=seed + 154
        ),
        stress=make_latest_value_retrieval_split(
            stress, records=48, query_occurrences=8, seed=seed + 155
        ),
        split_descriptions=(
            "12 records / 3 occurrences",
            "24 records / 5 occurrences",
            "48 records / 8 occurrences",
        ),
    )


RO_TYPE_CLS = 1
RO_TYPE_A = 2
RO_TYPE_B = 3
RO_TYPE_DISTRACTOR = 4
RO_DISTRACTOR_SYMBOLS = 8


def make_relative_order_split(
    count: int,
    *,
    item_count: int,
    seed: int,
) -> TensorDataset:
    count -= count % 2
    rng = random.Random(seed)
    tokens = torch.zeros(count, 1 + item_count, FIELD_COUNT, dtype=torch.long)
    labels = torch.empty(count, dtype=torch.long)

    for row in range(count):
        label = row % 2
        first, second = sorted(rng.sample(range(item_count), 2))
        if label == 1:
            a_position, b_position = first, second
        else:
            b_position, a_position = first, second
        sequence = [
            (RO_TYPE_DISTRACTOR, rng.randrange(1, RO_DISTRACTOR_SYMBOLS + 1), 0)
            for _ in range(item_count)
        ]
        sequence[a_position] = (RO_TYPE_A, 0, 0)
        sequence[b_position] = (RO_TYPE_B, 0, 0)
        tokens[row, 0] = torch.tensor((RO_TYPE_CLS, 0, 0))
        tokens[row, 1:] = torch.tensor(sequence)
        labels[row] = label

    order = torch.randperm(count, generator=torch.Generator().manual_seed(seed + 97))
    return TensorDataset(tokens.index_select(0, order), labels.index_select(0, order))


def build_relative_order(
    counts: tuple[int, int, int, int, int],
    *,
    seed: int,
) -> TaskBundle:
    train, val, test, long, stress = counts
    return TaskBundle(
        name="relative_order",
        description="Decide whether marker A occurs before marker B.",
        field_sizes=(5, RO_DISTRACTOR_SYMBOLS + 1, 1),
        num_classes=2,
        chance_accuracy=0.5,
        train=make_relative_order_split(train, item_count=24, seed=seed + 401),
        val=make_relative_order_split(val, item_count=24, seed=seed + 402),
        test=make_relative_order_split(test, item_count=24, seed=seed + 403),
        long=make_relative_order_split(long, item_count=64, seed=seed + 404),
        stress=make_relative_order_split(stress, item_count=128, seed=seed + 405),
        split_descriptions=("24 items", "64 items", "128 items"),
    )


GM_TYPE_CLS = 1
GM_TYPE_BIT = 2


def make_global_majority_split(
    count: int,
    *,
    item_count: int,
    seed: int,
) -> TensorDataset:
    count -= count % 2
    if item_count % 2 == 0:
        raise ValueError("item_count must be odd")

    rng = random.Random(seed)
    tokens = torch.zeros(count, 1 + item_count, FIELD_COUNT, dtype=torch.long)
    labels = torch.empty(count, dtype=torch.long)
    half = item_count // 2
    max_margin = max(1, item_count // 8)

    for row in range(count):
        label = row % 2
        margin = rng.randint(1, max_margin)
        ones = half + margin if label == 1 else half + 1 - margin
        ones = max(0, min(item_count, ones))
        bits = [1] * ones + [0] * (item_count - ones)
        rng.shuffle(bits)
        tokens[row, 0] = torch.tensor((GM_TYPE_CLS, 0, 0))
        tokens[row, 1:] = torch.tensor(
            [(GM_TYPE_BIT, bit + 1, 0) for bit in bits]
        )
        labels[row] = label

    order = torch.randperm(count, generator=torch.Generator().manual_seed(seed + 97))
    return TensorDataset(tokens.index_select(0, order), labels.index_select(0, order))


def build_global_majority(
    counts: tuple[int, int, int, int, int],
    *,
    seed: int,
) -> TaskBundle:
    train, val, test, long, stress = counts
    return TaskBundle(
        name="global_majority",
        description="Classify a binary majority near the decision boundary.",
        field_sizes=(3, 3, 1),
        num_classes=2,
        chance_accuracy=0.5,
        train=make_global_majority_split(train, item_count=15, seed=seed + 501),
        val=make_global_majority_split(val, item_count=15, seed=seed + 502),
        test=make_global_majority_split(test, item_count=15, seed=seed + 503),
        long=make_global_majority_split(long, item_count=63, seed=seed + 504),
        stress=make_global_majority_split(stress, item_count=127, seed=seed + 505),
        split_descriptions=("15 bits", "63 bits", "127 bits"),
    )


BL_TYPE_CLS = 1
BL_TYPE_OPERATOR = 2
BL_TYPE_NUMBER = 3
BL_TYPE_CLOSE = 4
BL_OP_AND = 1
BL_OP_OR = 2
BL_OP_XOR = 3


def apply_boolean_operator(operator: int, left: int, right: int) -> int:
    if operator == BL_OP_AND:
        return left & right
    if operator == BL_OP_OR:
        return left | right
    if operator == BL_OP_XOR:
        return left ^ right
    raise ValueError(f"unknown operator: {operator}")


def generate_boolean_expression(
    rng: random.Random,
    *,
    depth: int,
    max_depth: int,
    stop_probability: float,
) -> tuple[list[tuple[int, int, int]], int]:
    if depth >= max_depth or (depth > 0 and rng.random() < stop_probability):
        value = rng.randrange(2)
        return [(BL_TYPE_NUMBER, value + 1, 0)], value

    operator = rng.choice((BL_OP_AND, BL_OP_OR, BL_OP_XOR))
    left_tokens, left_value = generate_boolean_expression(
        rng,
        depth=depth + 1,
        max_depth=max_depth,
        stop_probability=stop_probability,
    )
    right_tokens, right_value = generate_boolean_expression(
        rng,
        depth=depth + 1,
        max_depth=max_depth,
        stop_probability=stop_probability,
    )
    return (
        [
            (BL_TYPE_OPERATOR, operator, 0),
            *left_tokens,
            *right_tokens,
            (BL_TYPE_CLOSE, 0, 0),
        ],
        apply_boolean_operator(operator, left_value, right_value),
    )


def make_boolean_listops_split(
    count: int,
    *,
    max_depth: int,
    seed: int,
) -> TensorDataset:
    count -= count % 2
    rng = random.Random(seed)
    quota = count // 2
    class_counts = [0, 0]
    sequences: list[list[tuple[int, int, int]]] = []
    labels: list[int] = []
    attempts = 0

    while len(sequences) < count:
        attempts += 1
        if attempts > count * 500:
            raise RuntimeError("could not build balanced Boolean ListOps split")
        expression, label = generate_boolean_expression(
            rng,
            depth=0,
            max_depth=max_depth,
            stop_probability=0.28,
        )
        if class_counts[label] >= quota:
            continue
        sequences.append([(BL_TYPE_CLS, 0, 0), *expression])
        labels.append(label)
        class_counts[label] += 1

    max_length = max(len(sequence) for sequence in sequences)
    tokens = torch.zeros(count, max_length, FIELD_COUNT, dtype=torch.long)
    label_tensor = torch.tensor(labels, dtype=torch.long)
    for row, sequence in enumerate(sequences):
        tokens[row, : len(sequence)] = torch.tensor(sequence)

    order = torch.randperm(count, generator=torch.Generator().manual_seed(seed + 97))
    return TensorDataset(tokens.index_select(0, order), label_tensor.index_select(0, order))


def build_boolean_listops(
    counts: tuple[int, int, int, int, int],
    *,
    seed: int,
) -> TaskBundle:
    train, val, test, long, stress = counts
    return TaskBundle(
        name="boolean_listops",
        description="Nested Boolean composition with unseen-depth generalization.",
        field_sizes=(5, 4, 1),
        num_classes=2,
        chance_accuracy=0.5,
        train=make_boolean_listops_split(train, max_depth=3, seed=seed + 301),
        val=make_boolean_listops_split(val, max_depth=3, seed=seed + 302),
        test=make_boolean_listops_split(test, max_depth=3, seed=seed + 303),
        long=make_boolean_listops_split(long, max_depth=4, seed=seed + 304),
        stress=make_boolean_listops_split(stress, max_depth=5, seed=seed + 305),
        split_descriptions=("depth <= 3", "unseen depth 4", "unseen depth 5"),
    )


TASK_BUILDERS: dict[
    str,
    Callable[[tuple[int, int, int, int, int]], TaskBundle],
] = {
    "associative_retrieval": build_associative_retrieval,
    "latest_value_retrieval": build_latest_value_retrieval,
    "relative_order": build_relative_order,
    "global_majority": build_global_majority,
    "boolean_listops": build_boolean_listops,
}


def build_task(
    name: str,
    counts: tuple[int, int, int, int, int],
    *,
    seed: int,
) -> TaskBundle:
    return TASK_BUILDERS[name](counts, seed=seed)


def sinusoidal_positions(
    length: int,
    dimension: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    position = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    frequency = torch.exp(
        torch.arange(0, dimension, 2, device=device, dtype=torch.float32)
        * (-math.log(10_000.0) / dimension)
    )
    encoding = torch.zeros(length, dimension, device=device, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(position * frequency)
    encoding[:, 1::2] = torch.cos(position * frequency)
    return encoding.to(dtype)


def move_query_to_last_valid(
    tokens: torch.Tensor,
    padding_mask: torch.Tensor,
) -> torch.Tensor:
    """Move the leading task query behind all records for causal readout."""
    lengths = (~padding_mask).sum(dim=1)
    if bool(lengths.le(0).any()):
        raise ValueError("every sequence must contain at least one valid token")
    positions = torch.arange(tokens.shape[1], device=tokens.device).unsqueeze(0)
    final = lengths.unsqueeze(1) - 1
    source = torch.where(
        positions < final,
        positions + 1,
        torch.where(positions == final, torch.zeros_like(positions), positions),
    )
    return tokens.gather(
        1,
        source.unsqueeze(-1).expand(-1, -1, tokens.shape[-1]),
    )


class TokenSequenceClassifier(nn.Module):
    def __init__(self, task: TaskBundle):
        super().__init__()
        if D_MODEL % FIELD_COUNT:
            raise ValueError("D_MODEL must be divisible by FIELD_COUNT")
        field_dim = D_MODEL // FIELD_COUNT
        self.field_embeddings = nn.ModuleList(
            nn.Embedding(size, field_dim, padding_idx=0)
            for size in task.field_sizes
        )
        self.input_projection = nn.Linear(D_MODEL, D_MODEL)
        self.classifier = nn.Linear(D_MODEL, task.num_classes)

    def encode(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        padding_mask = tokens[..., 0].eq(PAD_TYPE)
        tokens = move_query_to_last_valid(tokens, padding_mask)
        fields = [
            embedding(tokens[..., index])
            for index, embedding in enumerate(self.field_embeddings)
        ]
        x = self.input_projection(torch.cat(fields, dim=-1))
        x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        hidden = self.encode(x, padding_mask)
        last = (~padding_mask).sum(dim=1).sub(1).clamp_min(0)
        pooled = hidden[
            torch.arange(hidden.shape[0], device=hidden.device),
            last,
        ]
        return self.classifier(pooled)


class AATSequenceClassifier(TokenSequenceClassifier):
    def __init__(self, task: TaskBundle, encoder_layers: int):
        super().__init__(task)
        cfg = AATConfig.from_sequence_length(
            int(task.train.tensors[0].shape[1]),
            d_model=D_MODEL,
            encoder_layers=encoder_layers,
            decoder_layers=DECODER_LAYERS,
            heads=HEADS,
            rays=RAYS,
            ffn_dim=AAT_FFN_DIM,
            chunk_size=CHUNK_SIZE,
            dropout=DROPOUT,
            kappa=KAPPA,
            score_clip=SCORE_CLIP,
            gradient_checkpointing=False,
        )
        self.sequence = AAT(cfg)

    def encode(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.sequence(x, padding_mask)


class TransformerSequenceClassifier(TokenSequenceClassifier):
    def __init__(self, task: TaskBundle, ffn_dim: int, layers: int):
        super().__init__(task)
        layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=HEADS,
            dim_feedforward=ffn_dim,
            dropout=DROPOUT,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.sequence = nn.TransformerEncoder(
            layer,
            num_layers=layers,
            norm=nn.LayerNorm(D_MODEL),
        )
        self.dropout = nn.Dropout(DROPOUT)

    def encode(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        positions = sinusoidal_positions(
            x.shape[1],
            x.shape[2],
            device=x.device,
            dtype=x.dtype,
        )
        x = self.dropout(x + positions.unsqueeze(0))
        causal_mask = torch.ones(
            x.shape[1],
            x.shape[1],
            device=x.device,
            dtype=torch.bool,
        ).triu(1)
        return self.sequence(
            x,
            mask=causal_mask,
            src_key_padding_mask=padding_mask,
        ).masked_fill(padding_mask.unsqueeze(-1), 0.0)


class RecurrentSequenceClassifier(TokenSequenceClassifier):
    def __init__(
        self,
        task: TaskBundle,
        kind: str,
        hidden_size: int,
        layers: int,
    ):
        super().__init__(task)
        recurrent_type = nn.RNN if kind == "rnn" else nn.LSTM
        options = {"nonlinearity": "tanh"} if kind == "rnn" else {}
        self.sequence = recurrent_type(
            input_size=D_MODEL,
            hidden_size=hidden_size,
            num_layers=layers,
            dropout=DROPOUT if layers > 1 else 0.0,
            batch_first=True,
            **options,
        )
        self.output_projection = nn.Linear(hidden_size, D_MODEL)
        self.output_norm = nn.LayerNorm(D_MODEL)

    def encode(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        hidden, _ = self.sequence(x)
        return self.output_norm(
            self.output_projection(hidden)
        ).masked_fill(padding_mask.unsqueeze(-1), 0.0)


def make_model(task: TaskBundle, spec: ModelSpec) -> nn.Module:
    if spec.kind == "aat":
        return AATSequenceClassifier(task, encoder_layers=spec.width)
    if spec.kind == "transformer":
        return TransformerSequenceClassifier(task, spec.width, spec.layers)
    if spec.kind in {"rnn", "lstm"}:
        return RecurrentSequenceClassifier(
            task,
            spec.kind,
            spec.width,
            spec.layers,
        )
    raise ValueError(f"unknown model kind: {spec.kind}")


def candidate_parameter_count(
    task: TaskBundle,
    kind: str,
    width: int,
    layers: int,
) -> int:
    temporary = ModelSpec("", kind, width, layers, 0, "")
    model = make_model(task, temporary)
    parameters = count_parameters(model)
    del model
    return parameters


def nearest_width(
    task: TaskBundle,
    kind: str,
    target: int,
    *,
    layers: int,
    minimum: int,
    maximum: int,
) -> tuple[int, int]:
    cache: dict[int, int] = {}

    def parameters(width: int) -> int:
        if width not in cache:
            cache[width] = candidate_parameter_count(task, kind, width, layers)
        return cache[width]

    low = minimum
    high = maximum
    while low <= high:
        middle = (low + high) // 2
        if parameters(middle) < target:
            low = middle + 1
        else:
            high = middle - 1

    candidates = {
        max(minimum, min(maximum, value))
        for value in (high - 1, high, low, low + 1)
    }
    width = min(candidates, key=lambda value: abs(parameters(value) - target))
    return width, parameters(width)


def resolve_model_specs(task: TaskBundle) -> list[ModelSpec]:
    specs: list[ModelSpec] = []

    for encoder_layers in ENCODER_LAYERS:
        name = f"aat_e{encoder_layers}_d{DECODER_LAYERS}"
        parameters = candidate_parameter_count(
            task,
            "aat",
            encoder_layers,
            DECODER_LAYERS,
        )
        specs.append(
            ModelSpec(
                name=name,
                kind="aat",
                width=encoder_layers,
                layers=DECODER_LAYERS,
                parameters=parameters,
                detail=f"encoder={encoder_layers}, decoder={DECODER_LAYERS}",
            )
        )

    target = specs[0].parameters
    transformer_ffn, transformer_parameters = nearest_width(
        task,
        "transformer",
        target,
        layers=TRANSFORMER_LAYERS,
        minimum=8,
        maximum=1_024,
    )
    specs.append(
        ModelSpec(
            name="transformer",
            kind="transformer",
            width=transformer_ffn,
            layers=TRANSFORMER_LAYERS,
            parameters=transformer_parameters,
            detail=f"layers={TRANSFORMER_LAYERS}, ffn={transformer_ffn}",
        )
    )

    for kind, maximum in (("rnn", 512), ("lstm", 384)):
        hidden, parameters = nearest_width(
            task,
            kind,
            target,
            layers=RECURRENT_LAYERS,
            minimum=8,
            maximum=maximum,
        )
        specs.append(
            ModelSpec(
                name=kind,
                kind=kind,
                width=hidden,
                layers=RECURRENT_LAYERS,
                parameters=parameters,
                detail=f"layers={RECURRENT_LAYERS}, hidden={hidden}",
            )
        )

    return specs


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    correct = 0
    total = 0
    for tokens, labels in loader:
        tokens = tokens.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        predictions = model(tokens).argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.numel()
    return correct / max(total, 1)


def train_one_model(
    *,
    task: TaskBundle,
    spec: ModelSpec,
    train_cfg: TrainConfig,
    device: torch.device,
    seed: int,
) -> BenchmarkResult:
    seed_everything(seed)
    model = make_model(task, spec).to(device)
    parameters = count_parameters(model)
    if parameters != spec.parameters:
        raise RuntimeError(f"parameter count changed for {spec.name}")
    loaders = {
        split: make_loader(
            getattr(task, split),
            batch_size=train_cfg.batch_size,
            shuffle=split == "train",
            device=device,
            seed=seed,
        )
        for split in ("train", "val", "test", "long", "stress")
    }
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
    )
    scheduler = make_scheduler(
        optimizer,
        total_steps=train_cfg.epochs * len(loaders["train"]),
        warmup_fraction=train_cfg.warmup_fraction,
    )

    best_state: dict[str, torch.Tensor] | None = None
    best_val = -1.0
    best_epoch = 0
    start = time.perf_counter()

    print("\n" + "=" * 132)
    print(
        f"{task.name} | {spec.name} | seed={seed} | "
        f"parameters={parameters:,}"
    )
    print("=" * 132)

    for epoch in range(1, train_cfg.epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_examples = 0

        for tokens, labels in loaders["train"]:
            tokens = tokens.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(tokens)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            optimizer.step()
            scheduler.step()

            batch = labels.numel()
            total_loss += loss.item() * batch
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_examples += batch

        train_loss = total_loss / max(total_examples, 1)
        train_accuracy = total_correct / max(total_examples, 1)
        val_accuracy = evaluate(model, loaders["val"], device)
        if val_accuracy > best_val:
            best_val = val_accuracy
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

        print(
            f"epoch={epoch:02d}/{train_cfg.epochs} "
            f"loss={train_loss:.4f} train={train_accuracy:.4f} "
            f"val={val_accuracy:.4f} best={best_val:.4f} "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )

    train_seconds = time.perf_counter() - start
    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state)
    test_accuracy = evaluate(model, loaders["test"], device)
    long_accuracy = evaluate(model, loaders["long"], device)
    stress_accuracy = evaluate(model, loaders["stress"], device)

    print(
        f"best_epoch={best_epoch} best_val={best_val:.4f} "
        f"test={test_accuracy:.4f} long={long_accuracy:.4f} "
        f"stress={stress_accuracy:.4f} train_seconds={train_seconds:.1f}"
    )

    result = BenchmarkResult(
        task=task.name,
        model=spec.name,
        seed=seed,
        parameters=parameters,
        best_epoch=best_epoch,
        best_val=best_val,
        test=test_accuracy,
        long=long_accuracy,
        stress=stress_accuracy,
        train_seconds=train_seconds,
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def grouped_results(
    results: list[BenchmarkResult],
) -> dict[tuple[str, str], list[BenchmarkResult]]:
    groups: dict[tuple[str, str], list[BenchmarkResult]] = {}
    for result in results:
        groups.setdefault((result.task, result.model), []).append(result)
    return groups


def print_final_tables(
    tasks: list[TaskBundle],
    model_order: list[str],
    results: list[BenchmarkResult],
) -> None:
    groups = grouped_results(results)
    print("\n" + "=" * 142)
    print("FINAL SEQUENCE-MODEL RESULTS ACROSS SEEDS")
    print("=" * 142)
    print(
        f"{'task':<25}{'model':<14}{'params':>12}{'val':>10}{'test':>10}"
        f"{'long':>10}{'stress':>10}{'test_sd':>11}{'test_norm':>12}"
        f"{'train_s':>11}"
    )

    for task in tasks:
        for model_name in model_order:
            rows = groups[(task.name, model_name)]
            print(
                f"{task.name:<25}{rows[0].model:<14}{rows[0].parameters:>12,}"
                f"{mean([row.best_val for row in rows]):>10.4f}"
                f"{mean([row.test for row in rows]):>10.4f}"
                f"{mean([row.long for row in rows]):>10.4f}"
                f"{mean([row.stress for row in rows]):>10.4f}"
                f"{std([row.test for row in rows]):>11.4f}"
                f"{normalized_above_chance(mean([row.test for row in rows]), task.chance_accuracy):>12.4f}"
                f"{mean([row.train_seconds for row in rows]):>11.1f}"
            )

    print("\n" + "=" * 142)
    print("MACRO NORMALIZED SCORE")
    print("=" * 142)
    macro: dict[str, dict[str, float]] = {}
    for model_name in model_order:
        values = {"test": [], "long": [], "stress": []}
        for task in tasks:
            rows = groups[(task.name, model_name)]
            for split in values:
                values[split].append(
                    normalized_above_chance(
                        mean([getattr(row, split) for row in rows]),
                        task.chance_accuracy,
                    )
                )
        macro[model_name] = {
            split: mean(scores)
            for split, scores in values.items()
        }
        print(
            f"{model_name:<14} "
            f"test={macro[model_name]['test']:.4f} "
            f"long={macro[model_name]['long']:.4f} "
            f"stress={macro[model_name]['stress']:.4f}"
        )

    baseline = model_order[0]
    print("\n" + "=" * 142)
    print(f"DELTA AGAINST {baseline}")
    print("=" * 142)
    for model_name in model_order[1:]:
        print(
            f"{model_name:<14} "
            f"test={macro[model_name]['test'] - macro[baseline]['test']:+.4f} "
            f"long={macro[model_name]['long'] - macro[baseline]['long']:+.4f} "
            f"stress={macro[model_name]['stress'] - macro[baseline]['stress']:+.4f}"
        )


def verify_forward_backward(
    task: TaskBundle,
    specs: list[ModelSpec],
    device: torch.device,
) -> None:
    tokens, labels = task.train.tensors
    tokens = tokens[:8].to(device)
    labels = labels[:8].to(device)

    for spec in specs:
        seed_everything(123)
        model = make_model(task, spec).to(device)
        logits = model(tokens)
        if logits.shape != (8, task.num_classes):
            raise RuntimeError(f"bad output shape for {spec.name}")
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        if not math.isfinite(float(loss.item())):
            raise RuntimeError(f"non-finite smoke loss for {spec.name}")
        if spec.kind == "aat":
            if len(model.sequence.encoder.layers) != spec.width:
                raise RuntimeError("AAT encoder depth does not match its configuration")
            if len(model.sequence.decoder.layers) != spec.layers:
                raise RuntimeError("AAT decoder depth does not match its configuration")
            for layer in model.sequence.decoder.layers:
                if layer._writer is not model.sequence.encoder.ordered_memory:
                    raise RuntimeError("AAT decoder layers must share the final memory writer")
        for parameter in model.parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise RuntimeError(f"non-finite gradient for {spec.name}")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    train_cfg = TrainConfig()

    print("\n" + "#" * 132)
    print("COMMON SEQUENCE-MODEL FIVE-TASK ATTENTION-CAPACITY BENCHMARK")
    print("#" * 132)
    print(f"device={device} seeds={list(SEEDS)}")
    print(
        f"AAT encoder_layers={list(ENCODER_LAYERS)} "
        f"decoder_layers={DECODER_LAYERS}"
    )
    print(
        f"d_model={D_MODEL} heads={HEADS} rays={RAYS} "
        f"aat_ffn_dim={AAT_FFN_DIM} chunk_size={CHUNK_SIZE}"
    )
    print(f"train_config={train_cfg}")
    print(f"tasks={list(ALL_TASKS)}")
    print(
        "causal layout: the original leading query/CLS token is moved to "
        "the last valid position before every model"
    )

    tasks: list[TaskBundle] = []
    base_data_seed = 17_000
    for index, name in enumerate(ALL_TASKS):
        task = build_task(
            name,
            train_cfg.counts,
            seed=base_data_seed + index * 10_000,
        )
        tasks.append(task)
        print("\n" + "-" * 132)
        print(f"TASK: {task.name}")
        print(task.description)
        print(
            f"chance={task.chance_accuracy:.4f} "
            f"train/test={task.split_descriptions[0]} "
            f"long={task.split_descriptions[1]} "
            f"stress={task.split_descriptions[2]}"
        )

    specs_by_task = {
        task.name: resolve_model_specs(task)
        for task in tasks
    }
    model_order = [spec.name for spec in specs_by_task[tasks[0].name]]
    for task in tasks:
        specs = specs_by_task[task.name]
        if [spec.name for spec in specs] != model_order:
            raise RuntimeError("resolved model order differs across tasks")
        target = specs[0].parameters
        print("\n" + "-" * 132)
        print(f"PARAMETER MATCH: {task.name} | target={target:,}")
        for spec in specs:
            difference = 100.0 * (spec.parameters - target) / target
            print(
                f"{spec.name:<14} params={spec.parameters:>9,} "
                f"delta={difference:+7.3f}% | {spec.detail}"
            )

    verify_forward_backward(
        tasks[0],
        specs_by_task[tasks[0].name],
        device,
    )
    print("\nAll included models passed forward/backward checks.")

    results: list[BenchmarkResult] = []
    for task_index, task in enumerate(tasks):
        for spec in specs_by_task[task.name]:
            for seed in SEEDS:
                run_seed = seed + task_index * 100_000
                results.append(
                    train_one_model(
                        task=task,
                        spec=spec,
                        train_cfg=train_cfg,
                        device=device,
                        seed=run_seed,
                    )
                )

    print_final_tables(tasks, model_order, results)


if __name__ == "__main__":
    main()
