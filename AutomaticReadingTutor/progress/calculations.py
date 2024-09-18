def calculate_accuracy(total_words, correct_words):
    """
    Calculate the accuracy based on user progress and detailed progress data.
    :return: Accuracy percentage
    """

    accuracy = (correct_words / total_words) * 100 if total_words > 0 else 0
    return accuracy


def calculate_level(accuracy):
    """
        Calculate the level based on user progress and detailed progress data.
        :return: Level
        """
    level = accuracy // 10
    return level
