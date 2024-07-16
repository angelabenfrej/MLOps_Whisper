from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path

@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path
    base_model_path: Path
    updated_base_model_path: Path
    gradient_accumulation_steps: int
    learning_rate: float
    warmup_steps: int 
    max_steps: int
    gradient_checkpointing: bool
    fp16: bool
    per_device_eval_batch_size: int
    predict_with_generate: bool
    generation_max_length: int
    save_steps: int
    eval_steps: int


@dataclass(frozen=True)
class TrainingConfig:
    root_dir: Path
    trained_model_path: Path
    updated_base_model_path: Path
    training_data: Path


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
        processor: Any
        decoder_start_token_id: int

        