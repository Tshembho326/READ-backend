
from django.db import models
from AutomaticReadingTutor import settings


class UserProgress(models.Model):
    user = models.OneToOneField(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    total_level = models.IntegerField(default=0)
    accuracy = models.FloatField(default=0.0)

    def __str__(self):
        return f"{self.user.first_name}'s Progress"


class DetailedProgress(models.Model):
    user_progress = models.ForeignKey(UserProgress, related_name='detailed_progress', on_delete=models.CASCADE)
    level = models.CharField(max_length=50)
    level_value = models.IntegerField()
    progress = models.FloatField()

    def __str__(self):
        return f"{self.level} - Level {self.level_value}"
