from django.contrib import admin
from .models import UserProgress


class UserProgressAdmin(admin.ModelAdmin):
    list_display = ('user', 'accuracy')
    search_fields = ('email',)
    list_filter = ('user',)


admin.site.register(UserProgress, UserProgressAdmin)