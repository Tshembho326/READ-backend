import base64

import torch
import subprocess
import os
import tempfile
import json

from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from django.http import JsonResponse
from pydub import AudioSegment
from difflib import SequenceMatcher
from django.views.decorators.csrf import csrf_exempt

from authentication.models import CustomUser
from progress.calculations import calculate_accuracy
from progress.models import UserProgress

import io
import numpy as np

# Load the Wav2Vec2 model and processor
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft")


def generate_audio_for_word(word):
    """
    Generate a WAV file for the given word using eSpeak and return the path to the file.
    """
    espeak_path = r"C:\Program Files\eSpeak NG\espeak-ng.exe"  # Path to eSpeak on your system
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_wav:
        wav_path = temp_wav.name

    try:
        # Generate audio for the word using espeak
        command = [espeak_path, '-w', wav_path, word]
        result = subprocess.run(command, capture_output=True, text=True)
        result.check_returncode()
    except subprocess.CalledProcessError as e:
        print(f"Error generating audio for word '{word}': {e}")
        os.remove(wav_path)
        raise

    return wav_path


def generate_speech(text, output_path):
    try:
        espeak_path = r"C:\Program Files\eSpeak NG\espeak-ng.exe"
        command = [espeak_path, '-w', output_path, text]
        result = subprocess.run(command, capture_output=True, text=True)
        print("stdout:", result.stdout)
        print("stderr:", result.stderr)
        result.check_returncode()
    except subprocess.CalledProcessError as e:
        print(f"Error generating speech: {e}")
        print("stdout:", e.stdout)
        print("stderr:", e.stderr)
        raise


def generate_speech_in_memory(text):
    """
    Generate speech audio for a given text using espeak-ng and return the audio data as bytes.
    """
    espeak_path = r"C:\Program Files\eSpeak NG\espeak-ng.exe"
    command = [espeak_path, '--stdout', text]

    # Use subprocess to capture the output of espeak directly as audio data
    result = subprocess.run(command, capture_output=True)

    if result.returncode != 0:
        raise Exception(f"Error generating speech: {result.stderr.decode()}")

    # Return the generated audio data as bytes
    return result.stdout


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
    Returns the missed words in English and their corresponding audio files as Base64 encoded data.
    """
    if request.method == 'POST':
        try:
            username = request.POST.get('email')
            user_audio = request.FILES.get('audio')
            user = CustomUser.objects.get(email=username)

            # Check if the audio file is valid
            if not user_audio:
                return JsonResponse({'error': 'No audio file provided'}, status=400)

            # Get the story from the database
            story_title = request.POST.get('title')
            if not story_title:
                return JsonResponse({'error': 'Story title not provided'}, status=400)

            # Convert and transcribe the user audio
            wav_path = convert_audio(user_audio)
            transcription = transcribe_audio(wav_path)  # Using wav2 to phonemes
            print("transcription", transcription)
            os.remove(wav_path)  # Clean up temporary file
            if transcription is None:
                return JsonResponse({'error': 'Error processing audio file'}, status=500)

            story_text, story_phonemes = get_final_text_and_phonemes()  # Getting the phonemes and words sent by the frontend
            print("story", story_phonemes)
            results, missed_word_indices = align_with_levenshtein(transcription, story_phonemes)
            missed_words = extract_missed_words(story_text, missed_word_indices)

            total_words = len(story_text.split())  # Total words in the story
            correct_words = total_words - len(missed_words)  # Correct words based on missed words
            accuracy = calculate_accuracy(total_words, correct_words)

            # Update UserProgress
            progress, created = UserProgress.objects.get_or_create(
                user=user,
                defaults={
                    'accuracy': accuracy,
                    'total_words': total_words,
                    'correct_words': correct_words
                }
            )
            if not created:
                progress.accuracy = accuracy
                progress.total_words = total_words
                progress.correct_words = correct_words
                progress.missed_words = missed_words
                progress.save()

            # Generate audio for each missed word and convert to Base64
            audio_files = {}
            for word in missed_words:
                wav_path = generate_audio_for_word(word)
                with open(wav_path, 'rb') as f:
                    audio_blob = f.read()  # Read the file as binary data
                    audio_base64 = base64.b64encode(audio_blob).decode('utf-8')
                    audio_files[word] = audio_base64
                os.remove(wav_path)  # Clean up temporary file

            # Prepare the response data
            response_data = {
                'missed_words': missed_words,
                'audio_files': audio_files,  # Base64 encoded audio data
                'total_words': total_words,
                'correct_words': correct_words,
            }

            return JsonResponse(response_data)

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Invalid request method'}, status=405)


# Global variable to accumulate lines
accumulated_lines = []
final_text = ""
final_phonemes = ""


@csrf_exempt
def convert_to_phonemes(request):
    global accumulated_lines, final_text, final_phonemes

    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            new_lines = data.get('lines', [])
            is_final = data.get('is_final', False)  # A flag to check if it's the last part

            # Append the new lines to the global accumulated lines list
            accumulated_lines.extend(new_lines)
            final_text = ' '.join(accumulated_lines)
            final_phonemes = ''.join(generate_phonemes(final_text))

            # Clear the accumulated lines after generating phonemes
            accumulated_lines = []

            return JsonResponse({'message': 'Lines received, waiting for final part.'})
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Invalid request method'}, status=400)


def generate_phonemes(text):
    try:
        espeak_path = r"C:\Program Files\eSpeak NG\espeak-ng.exe"
        result = subprocess.run(
            [espeak_path, '-q', '--ipa=1', '-ven-za', text],
            capture_output=True,
            text=True,
            encoding='utf-8'
        )
        phonemes = result.stdout.strip()
        phonemes = phonemes.replace('_', ' ')
        phonemes = phonemes.replace('\'', '')
        return phonemes
    except Exception as e:
        print(f"Error generating phonemes with eSpeak: {e}")
        return None


# New function to retrieve the full text and phonemes
def get_final_text_and_phonemes():
    global final_text, final_phonemes

    if final_text and final_phonemes:
        # Return both the final concatenated text and phonemes
        return final_text, final_phonemes
    else:
        return None, None
