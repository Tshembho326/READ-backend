from django.db import models
from AutomaticReadingTutor import settings
from authentication.models import CustomUser


class UserProgress(models.Model):
    user = models.ForeignKey(CustomUser, on_delete=models.CASCADE)
    total_words = models.IntegerField(default=None)
    correct_words = models.IntegerField(default=None)
    accuracy = models.FloatField(null=True, blank=True, default=None)  # Optional: You can calculate it later

