
from src.whisper.config.configuration import ConfigurationManager
from src.whisper.components.model_evaluation_mlflow import Evaluation
from src.whisper import logger


STAGE_NAME= "Model Evaluation"


class ModelEvaluationPipeline:
    def __init___(self):
        pass
    def main(self):  
        config = ConfigurationManager()
        eval_config = config.get_evaluation_config()
        evaluation = Evaluation(eval_config)
        evaluation.load_data()
        wer_score = evaluation.compute_wer()
        evaluation.save_score(wer_score)
        evaluation.log_into_mlflow(wer_score)
    

if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelEvaluationPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e