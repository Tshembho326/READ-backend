from django.http import JsonResponse
from django.shortcuts import get_object_or_404
from .models import UserProgress
from .calculations import calculate_level
from authentication.models import CustomUser


def get_progress(request):
    """
    Retrieve the user's progress accuracy from the database using the user's email and return it as JSON.
    """
    # Find the user by email
    email = request.GET.get('email')

    if not email:
        return JsonResponse({'error': 'Email parameter is required'}, status=400)

    # Handle case where the user is not found
    try:
        user = CustomUser.objects.get(email=email)
    except CustomUser.DoesNotExist:
        return JsonResponse({'error': 'User not found'}, status=404)

    # Get the UserProgress for the found user
    progress = get_object_or_404(UserProgress, user=user)

    level = calculate_level(progress.accuracy)

    # Prepare the response data
    data = {
        'user_id': progress.user.id,
        'accuracy': progress.accuracy,
        'total_words': progress.total_words,
        'correct_words': progress.correct_words,
        'level': level,
    }

    return JsonResponse(data)
