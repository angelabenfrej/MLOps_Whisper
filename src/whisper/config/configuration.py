from src.whisper.constants import *
from src.whisper.utils.common import read_yaml, create_directories
from src.whisper.entity.config_entity import DataIngestionConfig
from src.whisper.entity.config_entity import PrepareBaseModelConfig

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config
    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model
        self.model_name = "openai/whisper-small"
        
        create_directories([config.root_dir])

        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            updated_base_model_path=Path(config.updated_base_model_path),
            gradient_accumulation_steps=self.params.gradient_accumulation_steps,
            learning_rate=self.params.learning_rate,
            warmup_steps=self.params.warmup_steps,
            max_steps=self.params.max_steps,
            gradient_checkpointing=self.params.gradient_checkpointing,
            fp16= self.params.fp16,
            per_device_eval_batch_size= self.params.per_device_eval_batch_size,
            predict_with_generate=self.params.predict_with_generate,
            generation_max_length=self.params.generation_max_length,
            save_steps=self.params.save_steps,
            eval_steps=self.params.save_steps
            
        )

        return prepare_base_model_config
      