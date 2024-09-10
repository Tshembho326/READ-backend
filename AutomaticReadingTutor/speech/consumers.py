import json
from channels.generic.websocket import AsyncWebsocketConsumer
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch

# Load model and processor (ideally this should be outside the class for efficiency)
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft")


class TranscriptionConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()

    async def receive(self, text_data):
        # Here text_data would contain the audio data
        transcription = await self.process_audio(text_data)  # Transcribe the audio
        await self.send(text_data=json.dumps({
            'transcription': transcription
        }))

    async def process_audio(self, audio_data):
        # This is where you use the Wav2Vec2Processor and model to transcribe
        input_values = processor(audio_data, return_tensors="pt", sampling_rate=16000).input_values

        # Perform transcription
        with torch.no_grad():
            logits = model(input_values).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)

        return transcription[0]  # Return the transcribed text
