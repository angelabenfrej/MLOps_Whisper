from src.whisper.config.configuration import ConfigurationManager
from src.whisper.components.model_trainer import Training
from src.whisper import logger


STAGE_NAME= "Model Training"

class ModelTrainingPipeline:
    def __init___(self):
        pass
    def main(self):   
        config = ConfigurationManager()
        training_config = config.get_training_config()
        training = Training(config=training_config)
        training.get_model()
        training.load_data()
        training.train()

if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e