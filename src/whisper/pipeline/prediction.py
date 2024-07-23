import torch
import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration

class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename
        self.model_path = "artifacts/training/model"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model(self):
        # Load the model
        model = WhisperForConditionalGeneration.from_pretrained(self.model_path, from_tf=False)
        
        # Load the processor
        try:
            processor = WhisperProcessor.from_pretrained(self.model_path)
        except OSError:
            # Handle the case where processor is not available
            processor = WhisperProcessor.from_pretrained(
                'openai/whisper-small', # or any other pre-trained processor name
                config=self.model_path
            )
        
        return model, processor
    def predict(self):
        audio, rate = torchaudio.load(self.filename)
        self.processor = WhisperProcessor.from_pretrained(
                'openai/whisper-small', # or any other pre-trained processor name
                config=self.model_path
            )
        self.model = WhisperForConditionalGeneration.from_pretrained(self.model_path, from_tf=False)

        # Resample audio to 16000 Hz
        if rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=rate, new_freq=16000)
            audio = resampler(audio)
            rate = 16000

        # Process audio
        input_features = self.processor(audio.squeeze().numpy(), sampling_rate=rate, return_tensors="pt").input_features.to(self.device)

        # Generate transcription
        with torch.no_grad():
            predicted_ids = self.model.generate(input_features)
            transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)

        return [{"transcription": transcription[0]}]