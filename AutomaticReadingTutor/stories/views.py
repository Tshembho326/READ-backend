from rest_framework.decorators import api_view
from rest_framework.response import Response
from .models import Story


@api_view(['GET'])
def get_story(request, title):
    try:
        story = Story.objects.get(title=title)
        story_data = {
            'id': story.id,
            'title': story.title,
            'content': story.content,
        }
        return Response(story_data)
    except Story.DoesNotExist:
        print(f"Story with title '{title}' not found.")
        return Response({'error': 'Story not found'}, status=404)
