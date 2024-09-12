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


def phonemes_to_text(phoneme_sequence):
    """
    Converts a sequence of phonemes back to English words using espeak.
    """
    try:
        espeak_path = r"C:\Program Files\eSpeak NG\espeak-ng.exe"
        result = subprocess.run(
            [espeak_path, '-q', '--ipa=1', '-ven-za', phoneme_sequence],
            capture_output=True,
            text=True,
            encoding='utf-8'
        )
        text_output = result.stdout.strip()
        return text_output
    except Exception as e:
        print(f"Error converting phonemes to text with eSpeak: {e}")
        return None


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
    Returns the missed words in English to the frontend.
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
                story_text = story.content         # The actual story text content (in English)
            except Story.DoesNotExist:
                return JsonResponse({'error': 'Story not found'}, status=404)

            # Transcribe the user audio
            transcription = transcribe_audio(user_audio)
            if transcription is None:
                return JsonResponse({'error': 'Error processing audio file'}, status=500)

            # Perform phoneme comparison and get mismatches
            mismatched_phonemes = []
            results, missed_word_indices = align_and_compare(transcription, story_phonemes, mismatched_phonemes)

            # Convert the missed phoneme indices to actual English words from the story
            missed_words = extract_missed_words(story_text, missed_word_indices)

            return JsonResponse({
                'transcription': transcription,
                'story_phonemes': story_phonemes,
                'results': results,
                'missed_words': missed_words  # Send the missed words back to the frontend as English
            })

        except Exception as e:
            print(f"Error in transcribe_and_compare: {e}")
            return JsonResponse({'error': 'Internal server error'}, status=500)

    return JsonResponse({'error': 'Invalid request method'}, status=400)


def align_and_compare(transcription_phonemes, story_phonemes, mismatched_phonemes):
    """
    Compare transcription phonemes with story phonemes, return comparison results,
    and collect mismatched phonemes. Also tracks which phonemes in the story were missed.
    """
    transcription_phonemes = transcription_phonemes.split()
    story_phonemes = story_phonemes.split()

    # Initialize empty list to store comparison results and word indices of mismatches
    comparison_results = []
    missed_word_indices = []

    i = 0
    j = 0
    len_trans = len(transcription_phonemes)
    len_story = len(story_phonemes)

    # Iterate over both phoneme sequences
    while i < len_trans and j < len_story:
        actual_phoneme = transcription_phonemes[i]
        expected_phoneme = story_phonemes[j]

        # Compare phonemes at the current position
        if actual_phoneme == expected_phoneme:
            comparison_results.append(f"Matched: {actual_phoneme}")
            i += 1
            j += 1
        else:
            # Store the mismatched phoneme from the story and track word index
            mismatched_phonemes.append(expected_phoneme)
            missed_word_indices.append(j)  # Track the index of the missed phoneme in the story

            # Check if the transcription phoneme matches the next story phoneme
            if j + 1 < len_story and actual_phoneme == story_phonemes[j + 1]:
                comparison_results.append(
                    f"Mismatch: expected '{expected_phoneme}', but matched '{actual_phoneme}' with next story phoneme")
                i += 1
                j += 2  # Skip over the next story phoneme as it's already matched
            # Check if the story phoneme matches the next transcription phoneme
            elif i + 1 < len_trans and transcription_phonemes[i + 1] == expected_phoneme:
                comparison_results.append(
                    f"Mismatch: expected '{expected_phoneme}', but matched next transcription phoneme '{transcription_phonemes[i + 1]}'")
                i += 2  # Skip over the next transcription phoneme as it's already matched
                j += 1
            else:
                comparison_results.append(f"Mismatch: expected '{expected_phoneme}', but got '{actual_phoneme}'")
                i += 1
                j += 1

    # If there are remaining phonemes in either sequence, add them to the results
    if i < len_trans:
        extra_actual = transcription_phonemes[i:]
        comparison_results.append(f"Extra in transcription: {' '.join(extra_actual)}")
    if j < len_story:
        extra_expected = story_phonemes[j:]
        comparison_results.append(f"Missing in transcription: {' '.join(extra_expected)}")
        mismatched_phonemes.extend(extra_expected)  # Add the remaining missing phonemes to mismatches
        missed_word_indices.extend(range(j, len_story))  # Track the remaining missed phonemes

    return comparison_results, missed_word_indices


def extract_missed_words(story_text, missed_word_indices):
    """
    Given the story text and the indices of the missed phonemes, return the actual missed words.
    Assumes that the phoneme list and text content correspond to one another.
    """
    words = story_text.split()  # Split the story into words
    missed_words = [words[i] for i in missed_word_indices if i < len(words)]  # Get words corresponding to missed phonemes
    return ' '.join(missed_words)
