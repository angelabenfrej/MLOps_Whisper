import os
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from src.whisper.config.configuration import TrainingConfig
from dataclasses import dataclass
from datasets import Dataset , Audio
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor, WhisperTokenizer, WhisperFeatureExtractor, Trainer, TrainingArguments


class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.processor = WhisperProcessor.from_pretrained("openai/whisper-small" ,language="en", task="transcribe" )
        self.tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", language="en", task="transcribe")
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
    def get_model(self):
        self.model = WhisperForConditionalGeneration.from_pretrained(self.config.updated_base_model_path)

    def load_data(self):
        import pandas as pd
        import numpy as np 
        import torchaudio
        # Load the dataset
        audio_folder = os.path.join(self.config.training_data, "Data_Whisper/en/Clips1")
        tsv_file = os.path.join(self.config.training_data, "Data_Whisper/en/validated1.tsv")
        data = pd.read_csv(tsv_file, sep='\t')
        
        # Limit to the first 10 entries
        data = data[:10]
        
        # Load and transform audio
        def load_and_transform_audio(path):
            try:
                file_path = os.path.join(audio_folder, str(path))
                print(f"Processing file: {file_path}")  
                waveform, sampling_rate = torchaudio.load(file_path)
                audio_array = waveform.numpy().astype(np.float32)
                audio_entry = {
                    'path': path,
                    'array': audio_array.flatten(),
                    'sampling_rate': 16000,
                }
                return audio_entry
            except Exception as e:
                print(f"Error processing audio file {path}: {str(e)}")
                return None

        data['audio'] = data['path'].apply(load_and_transform_audio)
        data = data.dropna(subset=['audio'])
        columns_to_remove = ['client_id', 'path', 'sentence_id', 'sentence_domain', 'up_votes', 'down_votes', 'age', 'gender', 'accents', 'variant', 'locale', 'segment']
        data = data.drop(columns=columns_to_remove)

        #Create Dataset object
        dataset = Dataset.from_pandas(data)     
        dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
        
        # Prepare the dataset
        def prepare_dataset(batch):
           # load and resample audio data from 48 to 16kHz
            audio = batch["audio"]

            # compute log-Mel input features from input audio array
            batch["input_features"] = self.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

            # encode target text to label ids
            batch["labels"] = self.tokenizer(batch["sentence"], padding=True).input_ids
            return batch

        dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names)

        self.train_dataset = dataset.train_test_split(test_size=0.2)["train"]
        self.eval_dataset = dataset.train_test_split(test_size=0.2)["test"]

    def train(self):
        training_args = TrainingArguments(
            output_dir=self.config.root_dir,
            per_device_train_batch_size=16,
            gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
            warmup_steps=500,
            gradient_checkpointing=True,
            fp16=True,
            evaluation_strategy="steps",
            per_device_eval_batch_size=8,
            save_steps=1000,
            eval_steps=1000,
            logging_steps=25,
            report_to=["tensorboard"],
            load_best_model_at_end=True,
            metric_for_best_model="wer",
            remove_unused_columns=False
        )

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


        data_collator = DataCollatorSpeechSeq2SeqWithPadding(
            processor=self.processor,
            decoder_start_token_id=self.model.config.decoder_start_token_id,
        )
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator
            
        )

        trainer.train()
        self.model.save_pretrained(self.config.trained_model_path)


