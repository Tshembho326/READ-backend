import torch
import subprocess
import os
import tempfile

from rest_framework.decorators import api_view
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from django.http import JsonResponse
from pydub import AudioSegment
from difflib import SequenceMatcher
from django.views.decorators.csrf import csrf_exempt

from progress.calculations import calculate_accuracy
from progress.models import UserProgress, DetailedProgress

from django.conf import settings
from stories.models import Story

import io
import numpy as np

# Load the Wav2Vec2 model and processor
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft")


def generate_speech(text, output_path):
    """
    Generate audio file for the given text using espeak.
    """
    try:
        espeak_path = r"C:\Program Files\eSpeak NG\espeak-ng.exe"
        command = [espeak_path, '-w', output_path, text]
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error generating speech: {e}")
        raise


def convert_audio(audio_file):
    """
    Convert the uploaded audio file to WAV format and return the path.
    """
    # Create a temporary file to store the WAV file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_wav:
        wav_path = temp_wav.name

    try:
        # Convert the uploaded file to WAV format using pydub
        audio = AudioSegment.from_file(audio_file)
        audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
        audio.export(wav_path, format="wav")
    except Exception as e:
        print(f"Error converting audio file: {e}")
        os.remove(wav_path)  # Clean up temporary file
        raise

    return wav_path


def transcribe_audio(file_path):
    """
    Transcribe the provided audio file using the Wav2Vec2 model.
    """
    try:
        # Load and process the audio file
        audio = AudioSegment.from_file(file_path, format="wav")
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
        print(f"Error in transcribing audio: {e}")
        return None


def levenshtein_distance(seq1, seq2):
    """
    Calculate the Levenshtein distance between two sequences.
    """
    n = len(seq1)
    m = len(seq2)

    # Create a matrix (n+1)x(m+1) for dynamic programming
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    # Initialize the matrix
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    # Compute the distance
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]  # No change if the characters match
            else:
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1  # Edit distance

    return dp[n][m]


def align_with_levenshtein(transcription_phonemes, story_phonemes):
    """
    Align transcription phonemes with story phonemes using Levenshtein distance.
    Return the mismatches and the word indices where the transcription differs.
    """
    transcription_phonemes = transcription_phonemes.split()
    story_phonemes = story_phonemes.split()

    # Calculate the Levenshtein distance
    distance = levenshtein_distance(transcription_phonemes, story_phonemes)

    # Find the differences using SequenceMatcher to provide alignment insights
    matcher = SequenceMatcher(None, transcription_phonemes, story_phonemes)
    mismatched_phonemes = []
    missed_word_indices = []
    comparison_results = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            comparison_results.append(f"Matched: {' '.join(transcription_phonemes[i1:i2])}")
        elif tag == 'replace':
            comparison_results.append(f"Mismatch: expected '{' '.join(story_phonemes[j1:j2])}', but got '{' '.join(transcription_phonemes[i1:i2])}'")
            mismatched_phonemes.extend(story_phonemes[j1:j2])
            missed_word_indices.extend(range(j1, j2))
        elif tag == 'delete':
            comparison_results.append(f"Missing in transcription: {' '.join(story_phonemes[j1:j2])}")
            mismatched_phonemes.extend(story_phonemes[j1:j2])
            missed_word_indices.extend(range(j1, j2))
        elif tag == 'insert':
            comparison_results.append(f"Extra in transcription: {' '.join(transcription_phonemes[i1:i2])}")

    return comparison_results, missed_word_indices


def extract_missed_words(story_text, missed_word_indices):
    """
    Extract the missed words from the story based on phoneme indices.
    """
    story_words = story_text.split()
    missed_words = set()

    for i in missed_word_indices:
        if i < len(story_words):
            missed_words.add(story_words[i])

    return list(missed_words)


@csrf_exempt
def transcribe_and_compare(request):
    """
    Handle the POST request to transcribe audio, generate phonemes, and compare it with the stored story phonemes.
    Returns the missed words in English and their corresponding audio to the frontend.
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
                story_phonemes = story.phoneme_content  # Use the pre-generated phonemes stored in the DB
                story_text = story.content  # The actual story text content (in English)
            except Story.DoesNotExist:
                return JsonResponse({'error': 'Story not found'}, status=404)

            # Convert and transcribe the user audio
            wav_path = convert_audio(user_audio)
            transcription = transcribe_audio(wav_path)
            os.remove(wav_path)  # Clean up temporary file
            if transcription is None:
                return JsonResponse({'error': 'Error processing audio file'}, status=500)

            # Perform phoneme comparison using Levenshtein distance and get mismatches
            results, missed_word_indices = align_with_levenshtein(transcription, story_phonemes)

            # Convert the missed phoneme indices to actual English words from the story
            missed_words = extract_missed_words(story_text, missed_word_indices)

            total_words = len(story_text.split())  # Total words in the story
            correct_words = total_words - len(missed_words)  # Correct words based on missed words

            # Generate audio files for missed words and create URLs
            audio_files = []
            audio_urls = []
            for word in missed_words:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio:
                    audio_path = temp_audio.name
                try:
                    generate_speech(word, audio_path)
                    audio_files.append(audio_path)
                    audio_urls.append(f"{settings.MEDIA_URL}{os.path.basename(audio_path)}")  # Create URL
                except Exception as e:
                    print(f"Error generating audio file for '{word}': {e}")

                    # Save the progress for the current user
                    user_progress, created = UserProgress.objects.get_or_create(user=request.user)

                    #
                    # Update or create the detailed progress for this story/level
                    detailed_progress, created = DetailedProgress.objects.get_or_create(
                        user_progress=user_progress,
                        level=9,
                        defaults={
                            'total_words': total_words,
                            'correct_words': correct_words,
                            'progress': (correct_words / total_words) * 100 if total_words > 0 else 0
                        }
                    )
                    #
                    if not created:
                        # If DetailedProgress already exists, update it
                        detailed_progress.total_words = total_words
                        detailed_progress.correct_words = correct_words
                        detailed_progress.progress = (correct_words / total_words) * 100 if total_words > 0 else 0
                        detailed_progress.save()

            # Prepare response with the transcription, results, missed words, and audio URLs
            response_data = {
                'transcription': transcription,
                'story_phonemes': story_phonemes,
                'results': results,
                'missed_words': missed_words,
                'audio_files': audio_urls  # Include URLs in response
            }

            # Clean up temporary audio files
            for file in audio_files:
                os.remove(file)

            return JsonResponse(response_data, status=200)

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Invalid request method'}, status=405)

