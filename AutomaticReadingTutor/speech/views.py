# from django.shortcuts import render
# from django.http import JsonResponse
# from .model_service import transcribe_audio
# from django.views.decorators.csrf import csrf_exempt
# import json
#
# @csrf_exempt
# def upload_audio(request):
#     if request.method == 'POST' and request.FILES.get('file'):
#         audio_file = request.FILES['file']
#         transcription = transcribe_audio(audio_file)
#         # return JsonResponse({'transcription': transcription})
#     return JsonResponse({'error': 'Invalid request'}, status=400)
