from django.db import models

class Story(models.Model):
    title = models.CharField(max_length=100, unique=True)  # Ensure titles are unique
    author = models.CharField(max_length=100)
    content = models.TextField()  # The actual story content
    difficulty = models.CharField(max_length=50)  # 'Easy', 'Medium', or 'Hard'

    def __str__(self):
        return self.title
