from django.db import models
from AutomaticReadingTutor import settings
from authentication.models import CustomUser


class UserProgress(models.Model):
    user = models.ForeignKey(CustomUser, on_delete=models.CASCADE)
    total_words = models.IntegerField(default=0)
    correct_words = models.IntegerField(default=0)
    missed_words = models.IntegerField(null=True, default=0)
    accuracy = models.FloatField(null=True, blank=True, default=0)
