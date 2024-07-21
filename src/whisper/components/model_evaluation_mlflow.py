from pydub import AudioSegment
from typing import Any, Dict, List, Union
import os
from datasets import Dataset, Audio
from dataclasses import dataclass
import evaluate
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import json
import torchaudio
from pathlib import Path
import mlflow
from urllib.parse import urlparse
from transformers import WhisperForConditionalGeneration, WhisperProcessor, WhisperTokenizer, WhisperFeatureExtractor
from src.whisper.entity.config_entity import EvaluationConfig
metric = evaluate.load("wer")

os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/benfredj.angela15/MLOps_Whisper.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"] = "benfredj.angela15"
os.environ["MLFLOW_TRACKING_PASSWORD"]= "9f92356bd6182df3299755fb8ff109d7d605bb39"



class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="en", task="transcribe")
        self.model = WhisperForConditionalGeneration.from_pretrained(self.config.path_of_model)
        self.tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", language="en", task="transcribe")
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
        self.metric = evaluate.load("wer")

        @dataclass
        class DataCollatorSpeechSeq2SeqWithPadding:
            processor: Any
            decoder_start_token_id: int

            def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
                input_features = [{"input_features": feature["input_features"]} for feature in features]
                batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

                label_features = [{"input_ids": feature["labels"]} for feature in features]
                labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
                labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

                if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
                    labels = labels[:, 1:]

                batch["labels"] = labels

                return batch
        
        self.data_collator = DataCollatorSpeechSeq2SeqWithPadding(
            processor=self.processor,
            decoder_start_token_id=self.model.config.decoder_start_token_id,
        )

    def load_model(self, path: Path):
        return WhisperForConditionalGeneration.from_pretrained(path)
    
    def load_data(self):
        import pandas as pd
        import numpy as np

        audio_folder = os.path.join(self.config.training_data, "Data_Whisper/en/Clips1")
        tsv_file = os.path.join(self.config.training_data, "Data_Whisper/en/validated1.tsv")
        data = pd.read_csv(tsv_file, sep='\t')
        data = data[:10]

        def load_and_transform_audio(path):
            try:
                file_path = os.path.join(audio_folder, str(path))
                waveform, sampling_rate = torchaudio.load(file_path)
                audio_array = waveform.numpy().astype(np.float32)
                audio_entry = {
                    'path': str(path),
                    'array': audio_array.flatten(),
                    'sampling_rate': 16000,
                }
                return audio_entry
            except Exception as e:
                print(f"Error processing file {path}: {str(e)}")
                return None

        data['audio'] = data['path'].apply(load_and_transform_audio)
        data = data.dropna(subset=['audio'])

        columns_to_remove = ['client_id', 'path', 'sentence_id', 'sentence_domain', 'up_votes', 'down_votes', 'age', 'gender', 'accents', 'variant', 'locale', 'segment']
        data = data.drop(columns=columns_to_remove)

        dataset = Dataset.from_pandas(data)     
        dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

        def prepare_dataset(batch):
            audio = batch["audio"]
            batch["input_features"] = self.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
            batch["labels"] = self.tokenizer(batch["sentence"], padding=True).input_ids
            return batch

        dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names)
        self.train_dataset = dataset.train_test_split(test_size=0.2)["train"]
        self.eval_dataset = dataset.train_test_split(test_size=0.2)["test"]

    def collate_fn(self, batch):
        input_features = [torch.tensor(item["input_features"]) for item in batch]
        labels = [torch.tensor(item["labels"]) for item in batch]

        input_features = pad_sequence(input_features, batch_first=True, padding_value=0)
        labels = pad_sequence(labels, batch_first=True, padding_value=self.processor.tokenizer.pad_token_id)

        return {"input_features": input_features, "labels": labels}

    def compute_wer(self):
        predictions = []
        references = []

        batch_size = 2
        eval_loader = DataLoader(self.eval_dataset, batch_size=batch_size, collate_fn=self.data_collator)

        for batch in eval_loader:
            input_features = batch["input_features"]
            labels = batch["labels"]

            try:
                generated_ids = self.model.generate(input_features)
                batch_predictions = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
                batch_references = [self.processor.decode(label, skip_special_tokens=True) for label in labels]

                predictions.extend(batch_predictions)
                references.extend(batch_references)
            except Exception as e:
                print(f"Error during batch processing: {e}")

        wer = self.metric.compute(predictions=predictions, references=references)
        return wer

    def save_score(self, wer_score):
        scores = {"wer": wer_score}
        with open("scores.json", "w") as f:
            json.dump(scores, f)

    def log_into_mlflow(self, score):
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        with mlflow.start_run() as run:
            mlflow.log_params(self.config.all_params)
            mlflow.log_metric("wer", score)
            run_id = run.info.run_id
            subpath = "world_1-1"
            run_uri = f'runs:/{run_id}/{subpath}'
            model_version = mlflow.register_model(run_uri, "WhisperModel")