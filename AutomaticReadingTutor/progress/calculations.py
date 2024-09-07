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

    total_reads = len(detailed_progress)
    correct_reads = sum(dp.progress for dp in detailed_progress)  # Adjust this based on actual logic
    accuracy = (correct_reads / total_reads) * 100 if total_reads > 0 else 0
    return accuracy
