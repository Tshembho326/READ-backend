from django.http import JsonResponse
from rest_framework.decorators import api_view

from progress.models import UserProgress


@api_view(['GET'])
def get_progress(request):
    """
    Retrieve the progress of the currently authenticated user.
    """
    user = request.user

    if not user.is_authenticated:
        return JsonResponse({'error': 'User not authenticated'}, status=401)

    try:
        user_progress = UserProgress.objects.get(user=user)
        progress_data = {
            'total_words': user_progress.total_words,
            'correct_words': user_progress.correct_words,
            'progress_percentage': user_progress.progress,
            'accuracy': user_progress.accuracy
        }
        return JsonResponse(progress_data)

    except UserProgress.DoesNotExist:
        return JsonResponse({'error': 'User progress not found'}, status=404)
