from django.contrib import admin
from django.urls import path
from authentication import views as auth
from speech import views as voice
from stories import views as story
from progress import views as prog

urlpatterns = [
    path('admin/', admin.site.urls),
    
    # Authentication paths 
    path('register/', auth.registerUser, name='register'),
    path('login/', auth.loginUser, name='login'),
    path('forgot-password/', auth.forgotPassword, name='forgot_password'),
    path('reset-password/<uidb64>/<token>/', auth.resetPassword, name='reset_password'),
    path('change-password/', auth.changePassword, name='change_password'),
    path('change-user-details/', auth.changeUserDetails, name="change_user_details"),
    path('logout/', auth.logout, name='logout'),

    # Speech paths
    path('transcribe/', voice.transcribe_and_compare, name='transcribe'),

    # library path
    path('stories/', story.get_story, name='stories'),
    path('stories/<str:title>/', story.get_story, name='stories'),

    # Progress path
    path('progress/', prog.get_progress, name='progress'),
]
