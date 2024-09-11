from django.db import models
import subprocess


def generate_phonemes(text):
    try:
        espeak_path = r"C:\Program Files\eSpeak NG\espeak-ng.exe"  # Full path to espeak
        result = subprocess.run(
            [espeak_path, '-q', '--ipa=1', '-ven-za', text],
            capture_output=True,
            text=True,
            encoding='utf-8'
        )
        phonemes = result.stdout.strip()
        phonemes = phonemes.replace('_', ' ')
        phonemes = phonemes.replace('\'', '')
        return phonemes
    except Exception as e:
        print(f"Error generating phonemes with eSpeak: {e}")
        return None


class Story(models.Model):
    title = models.CharField(max_length=100, unique=True)  # Ensure titles are unique
    author = models.CharField(max_length=100)
    content = models.TextField()  # The actual story content
    phoneme_content = models.TextField(blank=True, null=True)
    difficulty = models.CharField(max_length=50)  # 'Easy', 'Medium', or 'Hard'

    def __str__(self):
        return self.title

    def save(self, *args, **kwargs):
        # Automatically convert content to phonemes before saving
        if self.content:  # Only generate phonemes if there's content
            self.phoneme_content = generate_phonemes(self.content)
        super().save(*args, **kwargs)
