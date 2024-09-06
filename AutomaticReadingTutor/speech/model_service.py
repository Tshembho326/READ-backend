# from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
# import torchaudio    # I am having troubles with installing Torch
# import io
#
# # Load model and processor
# processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft")
# model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft")
#
# def transcribe_audio(audio_file):
#     # Read and process the audio file
#     # audio_array = audio_file.read()
#     input_values = processor(audio_array, return_tensors="pt", sampling_rate=16000).input_values
#
#     # Retrieve logits
#     with torch.no_grad():
#         logits = model(input_values).logits
#
#     # Decode the logits to text
#     predicted_ids = torch.argmax(logits, dim=-1)
#     transcription = processor.batch_decode(predicted_ids)
#     decoded_transcription = transcription[0]
#
#     return decoded_transcription