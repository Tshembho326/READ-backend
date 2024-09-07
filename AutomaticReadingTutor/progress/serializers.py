from rest_framework import serializers
from .models import UserProgress, DetailedProgress


class DetailedProgressSerializer(serializers.ModelSerializer):
    class Meta:
        model = DetailedProgress
        fields = ['level', 'level_value', 'progress']


class UserProgressSerializer(serializers.ModelSerializer):
    detailed_progress = DetailedProgressSerializer(many=True, read_only=True)

    class Meta:
        model = UserProgress
        fields = ['total_level', 'accuracy', 'detailed_progress']
