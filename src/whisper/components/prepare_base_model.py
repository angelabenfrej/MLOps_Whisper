from dataclasses import dataclass
from pathlib import Path
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from src.whisper.entity.config_entity import PrepareBaseModelConfig

class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config
    
    def get_base_model(self):
        self.model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
        self.processor = WhisperProcessor.from_pretrained("openai/whisper-small")
        self.save_model(self.config.base_model_path, self.model)

    def save_model(self, path: Path, model):
        model.save_pretrained(path)
        print(f"Model saved to {path}")

    def _prepare_full_model(self, model, processor, **kwargs):
        
        return model, processor
    

    def update_base_model(self):
        # This method would be used to update the base model if needed
        self.full_model, self.processor = self._prepare_full_model(
            model=self.model,
            processor=self.processor,
            learning_rate=self.config.learning_rate,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            warmup_steps=self.config.warmup_steps,
            max_steps=self.config.max_steps,
            gradient_checkpointing=self.config.gradient_checkpointing,
            fp16=self.config.fp16,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            predict_with_generate=self.config.predict_with_generate,
            generation_max_length=self.config.generation_max_length,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps
        )
        self.save_model(self.config.updated_base_model_path, self.full_model)

    @staticmethod
    def save_model(path: str, model):
        model.save_pretrained(path)
        print(f"Model saved to {path}")

