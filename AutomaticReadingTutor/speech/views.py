import torch
import subprocess
from rest_framework.decorators import api_view
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from django.http import JsonResponse
from pydub import AudioSegment
from stories.models import Story
import io
import numpy as np
from django.views.decorators.csrf import csrf_exempt

# Load the Wav2Vec2 model and processor
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft")


def transcribe_audio(file):
    """
    Transcribe the provided audio file using the Wav2Vec2 model.
    """
    try:
        # Ensure the file is not None and is in WAV format
        if file is None:
            raise ValueError("No file provided")

        # Load and process the audio file
        audio = AudioSegment.from_file(file, format="wav")
        audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)

        # Convert audio to numpy array
        audio_bytes = io.BytesIO()
        audio.export(audio_bytes, format="wav")
        audio_bytes.seek(0)
        audio_input = np.frombuffer(audio_bytes.read(), np.int16).astype(np.float32) / 32768.0

        # Prepare input for the model
        input_values = processor(audio_input, return_tensors="pt", sampling_rate=16000).input_values

        # Perform transcription
        with torch.no_grad():
            logits = model(input_values).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)

        return transcription[0]

    except Exception as e:
        # Handle and log exceptions
        print(f"Error in transcribing audio: {e}")
        return None


@csrf_exempt
def transcribe_and_compare(request):
    """
    Handle the POST request to transcribe audio, generate phonemes, and compare it with the stored story phonemes.
    """
    if request.method == 'POST':
        try:
            user_audio = request.FILES.get('audio')

            # Check if the audio file is valid
            if not user_audio:
                return JsonResponse({'error': 'No audio file provided'}, status=400)

            # Get the story from the database
            story_title = request.POST.get('title')
            if not story_title:
                return JsonResponse({'error': 'Story title not provided'}, status=400)

            try:
                story = Story.objects.get(title=story_title)
                story_phonemes = story.phoneme_content  # Use the pre-generated phonemes that are stored in the DB
            except Story.DoesNotExist:
                return JsonResponse({'error': 'Story not found'}, status=404)

            # Transcribe the user audio
            transcription = transcribe_audio(user_audio)
            if transcription is None:
                return JsonResponse({'error': 'Error processing audio file'}, status=500)

            return JsonResponse({
                'transcription': transcription,
                'story_phonemes': story_phonemes,
            })

        except Exception as e:
            print(f"Error in transcribe_and_compare: {e}")
            return JsonResponse({'error': 'Internal server error'}, status=500)

    return JsonResponse({'error': 'Invalid request method'}, status=400)
