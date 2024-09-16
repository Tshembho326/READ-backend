from django.http import JsonResponse
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.authentication import SessionAuthentication, BasicAuthentication
from progress.calculations import calculate_accuracy
from progress.models import UserProgress


@api_view(['GET'])
def get_progress(request):
    """
    Retrieve the progress of the currently authenticated user, including updated accuracy.
    """
    user = request.user

    if not user.is_authenticated:
        return JsonResponse({'error': 'User not authenticated'}, status=401)

    try:
        # Get the user's overall progress
        user_progress = UserProgress.objects.get(user=user)

        # Calculate the accuracy using the detailed progress data
        accuracy = calculate_accuracy(user_progress)

        # Update the user's overall accuracy
        user_progress.accuracy = accuracy
        user_progress.save()

        # Prepare the detailed progress data
        detailed_progress = user_progress.detailed_progress.all()
        progress_data = {
            'total_level': user_progress.total_level,
            'accuracy': accuracy,
            'detailed_progress': [
                {
                    'level': dp.level,
                    'total_words': dp.total_words,
                    'correct_words': dp.correct_words,
                    'progress_percentage': dp.progress
                } for dp in detailed_progress
            ]
        }

        return JsonResponse(progress_data, status=200)

    except UserProgress.DoesNotExist:
        return JsonResponse({'error': 'User progress not found'}, status=404)


def update_user_progress(user):
    """
    Update the UserProgress after the user has completed a story or level.
    :param user: The user who has completed the reading
    """
    try:
        # Get the user's progress
        user_progress = UserProgress.objects.get(user=user)

        # Calculate the new accuracy
        new_accuracy = calculate_accuracy(user_progress)

        # Update the accuracy in the UserProgress model
        user_progress.accuracy = new_accuracy
        user_progress.save()

    except UserProgress.DoesNotExist:
        # Handle the case where the user progress record doesn't exist
        print(f"User progress for {user.username} not found.")
