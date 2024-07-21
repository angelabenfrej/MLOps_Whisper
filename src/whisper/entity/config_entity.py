from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Dict, List, Union
import torch


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
        def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
                    # split inputs and labels since they have to be of different lengths and need different padding methods
                    # first treat the audio inputs by simply returning torch tensors
                    input_features = [{"input_features": feature["input_features"]} for feature in features]
                    batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

                    # get the tokenized label sequences
                    label_features = [{"input_ids": feature["labels"]} for feature in features]
                    # pad the labels to max length
                    labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

                    # replace padding with -100 to ignore loss correctly
                    labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

                    # if bos token is appended in previous tokenization step,
                    # cut bos token here as it's append later anyways
                    if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
                        labels = labels[:, 1:]

                    batch["labels"] = labels

                    return batch
        

@dataclass(frozen=True)
class EvaluationConfig:
    path_of_model: Path
    training_data: Path
    all_params: dict
    mlflow_uri: str

            