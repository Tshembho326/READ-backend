from .models import DetailedProgress


def calculate_accuracy(user_progress):
    """
    Calculate the accuracy based on user progress and detailed progress data.
    :param user_progress: The UserProgress instance
    :return: Accuracy percentage
    """
    detailed_progress = DetailedProgress.objects.filter(user_progress=user_progress)

    if not detailed_progress:
        return 0.0

    total_words = sum(dp.total_words for dp in detailed_progress)
    correct_words = sum(dp.correct_words for dp in detailed_progress)

    accuracy = (correct_words / total_words) * 100 if total_words > 0 else 0
    return accuracy

