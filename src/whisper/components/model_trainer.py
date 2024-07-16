import os
from pydub import AudioSegment
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from src.whisper.config.configuration import TrainingConfig
from transformers import Trainer, TrainingArguments, WhisperForConditionalGeneration, WhisperProcessor
from dataclasses import dataclass
from datasets import Dataset , Audio
import torch






AudioSegment.ffmpeg = "ffmpeg"
class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.processor = WhisperProcessor.from_pretrained("openai/whisper-small")
    def get_model(self):
        self.model = WhisperForConditionalGeneration.from_pretrained(self.config.updated_base_model_path)
        self.model.generation_config.language = "english"
        self.model.generation_config.task = "transcribe"
        self.model.generation_config.forced_decoder_ids = None

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
                file_path = os.path.join(audio_folder, path)
                audio= AudioSegment.from_file(file_path)
                audio_array = np.array(audio.get_array_of_samples()).astype(np.float32)
                audio_entry = {
                    'path': path,
                    'array': audio_array.flatten(),
                    'sampling_rate': 16000,
                }
                return audio_entry
            except Exception as e:
                return None

        data['audio'] = data['path'].apply(load_and_transform_audio)
        data = data.dropna(subset=['audio'])
        audio_entries = []
        for index, row in data.iterrows():
            audio_dict = load_and_transform_audio(row)
            if audio_dict:
                audio_entries.append(audio_dict)
            else:
        # Handle cases where audio loading fails or path is NaN
                audio_entries.append(None)
        # Remove unnecessary columns
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
            batch["input_features"] = self.processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

            # encode target text to label ids
            batch["labels"] = self.processor.tokenizer(batch["sentence"], padding=True).input_ids
            return batch

        dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names)

        self.train_dataset = dataset.train_test_split(test_size=0.2)["train"]
        self.eval_dataset = dataset.train_test_split(test_size=0.2)["test"]



    def compute_metrics(self, pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = self.processor.tokenizer.pad_token_id

        pred_str = self.processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = self.processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer = 100 * self.metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    def train(self):
        training_args = TrainingArguments(
            output_dir=self.config.root_dir,
            per_device_train_batch_size=16,
            gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
            learning_rate=1e-5,
            warmup_steps=500,
            max_steps=3,
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
            tokenizer=self.processor.feature_extractor,
            data_collator=data_collator
            
        )

        trainer.train()
        self.model.save_pretrained(self.config.trained_model_path)