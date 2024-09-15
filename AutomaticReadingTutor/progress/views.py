from django.shortcuts import get_object_or_404
from django.http import JsonResponse
from django.views.decorators.http import require_GET
from django.contrib.auth.decorators import login_required
from .models import UserProgress
from .serializers import UserProgressSerializer
from .calculations import calculate_accuracy


@login_required
@require_GET
def get_progress(request):
    # Get the current logged-in user
    user = request.user

    # Retrieve user progress data or return a 404 if not found
    user_progress = get_object_or_404(UserProgress, user=user)

    # Calculate accuracy only if necessary (for example, if progress changed)
    new_accuracy = calculate_accuracy(user_progress)

    # Update accuracy only if it has changed
    if user_progress.accuracy != new_accuracy:
        user_progress.accuracy = new_accuracy
        user_progress.save()

    # Serialize the user progress data
    serializer = UserProgressSerializer(user_progress)

    # Return JSON response with the serialized data
    return JsonResponse(serializer.data, status=200)

