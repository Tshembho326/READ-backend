from rest_framework.decorators import api_view
from rest_framework.response import Response
from .models import Story


@api_view(['GET'])
def get_story(request, title):
    print(f"Received title: {title}")

    try:
        # Fetch the story using the title
        story = Story.objects.get(title=title)
        print(f"Found story: {story}")

        # Prepare the story data
        story_data = {
            'id': story.id,
            'title': story.title,
            'content': story.content,
        }

        # Return the story data
        return Response(story_data)

    except Story.DoesNotExist:
        print(f"Story with title '{title}' not found.")
        return Response({'error': 'Story not found'}, status=404)
