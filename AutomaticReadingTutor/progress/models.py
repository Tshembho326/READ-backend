from django.db import models
from AutomaticReadingTutor import settings


class UserProgress(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    total_words = models.IntegerField(default=None)
    correct_words = models.IntegerField(default=None)
    accuracy = models.FloatField(null=True, blank=True, default=None)  # Optional: You can calculate it later

    def save(self, *args, **kwargs):
        if self.total_words and self.correct_words:
            self.accuracy = (self.correct_words / self.total_words) * 100
        super().save(*args, **kwargs)
