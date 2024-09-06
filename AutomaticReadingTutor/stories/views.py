from rest_framework.decorators import api_view
from rest_framework.response import Response
from .models import Story


@api_view(['GET'])
def get_story(request):
    # Extract title from query parameters
    title = request.GET.get('title')

    if not title:
        return Response({'error': 'Title parameter is required'}, status=400)

    try:
        story = Story.objects.get(title=title)

        # Assuming Story model has these fields
        story_data = {
            'id': story.id,
            'title': story.title,
            'content': story.content,
            'difficulty': story.difficulty,
        }

        return Response(story_data)

    except Story.DoesNotExist:
        return Response({'error': 'Story not found'}, status=404)
