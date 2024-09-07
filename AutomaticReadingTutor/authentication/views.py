from django.contrib.auth import authenticate, get_user_model
from django.http import JsonResponse
from django.core.mail import send_mail
from django.utils.http import urlsafe_base64_encode, urlsafe_base64_decode
from django.utils.encoding import force_bytes, force_str
from django.utils.crypto import get_random_string
from django.contrib.auth.tokens import default_token_generator, PasswordResetTokenGenerator
from django.conf import settings
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from knox.models import AuthToken
from .serializers import UserSerializer
from rest_framework.authtoken.serializers import AuthTokenSerializer
from django.views.decorators.csrf import csrf_exempt
from .models import CustomUser


import json

# Get the custom user model for your application
User = get_user_model()


# ---------------- Register User ----------------
@csrf_exempt
@api_view(['POST'])
def registerUser(request):
    """
    Registers a new user by validating the data sent in the request.
    If the data is valid, it saves the new user and returns a success message.
    Otherwise, it returns a 400 error with the validation errors.
    """
    if request.method == 'POST':
        serializer = UserSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(
                {'message': 'User created successfully'},
                status=status.HTTP_201_CREATED
            )
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


# ---------------- Login User ----------------
@api_view(['POST'])
def loginUser(request):
    """
    Logs in a user by validating the provided credentials (email and password).
    If valid, it authenticates the user and generates a token for them.
    The token and user information are returned if successful, 
    otherwise an error message is returned.
    """
    serializer = AuthTokenSerializer(data=request.data)
    if serializer.is_valid():
        user = serializer.validated_data['user']

        # Authenticate user using email and password
        user = authenticate(username=user.email, password=serializer.validated_data['password'])

        if user is not None:
            # Generate an authentication token for the user
            _, token = AuthToken.objects.create(user)
            return JsonResponse({
                'user': {
                    'email': user.email,
                    'first_name': user.first_name,
                    'last_name': user.last_name
                },
                'token': token
            })
        return JsonResponse({'message': 'Invalid credentials'}, status=status.HTTP_400_BAD_REQUEST)
    return JsonResponse(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


# ---------------- Forgot Password ----------------
@api_view(['POST'])
def forgotPassword(request):
    """
    Sends a password reset email to the user if the provided email exists.
    Generates a unique token and user ID, which are used to create a reset link.
    If the user does not exist, it returns an error message.
    """
    email = request.data.get('username')
    if email:
        try:
            # Find user by email
            user = User.objects.get(email=email)

            # Generate token and user ID
            token = default_token_generator.make_token(user)
            uid = urlsafe_base64_encode(force_bytes(user.pk))

            # Build password reset link
            reset_link = f"http://localhost:3000/reset-password/{uid}/{token}/"

            # Send password reset email
            send_mail(
                'Password Reset Request',
                f'Hi {user.first_name},\n\nYou requested a password reset for READ. Click the link below to reset your password:\n{reset_link}\n\nIf you did not make this request, please ignore this email.',
                settings.EMAIL_HOST_USER,
                [user.email],
                fail_silently=False,
            )
            return JsonResponse({'message': 'Password reset for READ email sent.'}, status=200)
        except User.DoesNotExist:
            return JsonResponse({'error': 'User with this email does not exist.'}, status=400)
    return JsonResponse({'error': 'Email field is required.'}, status=400)


# ---------------- Reset Password ----------------
@api_view(['POST'])
def resetPassword(request, uidb64, token):
    """
    Resets the user's password using the provided UID and token.
    If the token is valid, it sets the new password for the user and saves it.
    If the token is invalid or an error occurs, it returns an error message.
    """
    password = request.data.get('password')

    try:
        # Decode the user ID and retrieve the user
        uid = force_str(urlsafe_base64_decode(uidb64))
        user = User.objects.get(pk=uid)

        # Verify the token
        token_generator = PasswordResetTokenGenerator()
        if token_generator.check_token(user, token):
            # Set new password and save user
            user.set_password(password)
            user.save()
            return JsonResponse({'message': 'Password reset successful'}, status=200)
        return JsonResponse({'error': 'Invalid token'}, status=400)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=400)


# ---------------- Change Password ----------------
@api_view(['POST'])
def changePassword(request):
    """
    Changes the user's password if the old password is correct.
    Authenticates the user using the old password and then updates it to the new password.
    Returns an error message if authentication fails or an error occurs.
    """
    if request.method == 'POST':
        try:
            # Parse JSON data from request body
            body = json.loads(request.body.decode('utf-8'))
            oldPassword = body.get('oldPassword')
            email = body.get('email')
            newPassword = body.get('newPassword')

            # Authenticate user with old password
            user = authenticate(email=email, password=oldPassword)
            if user is not None:
                # Set new password and save user
                user.set_password(newPassword)
                user.save()
                return JsonResponse({'message': 'Password changed successfully.'}, status=200)
            return JsonResponse({'message': 'Incorrect old password.'}, status=400)
        except json.JSONDecodeError:
            return JsonResponse({'message': 'Invalid JSON data'}, status=400)
    return JsonResponse({'message': 'Method not allowed'}, status=405)


# ---------------- Logout ----------------
@api_view(['POST'])
def logout(request):
    """
    Logs the user out by deleting all tokens associated with the user.
    The user's email is passed in the request body and all tokens for that email are deleted.
    """
    if request.method == 'POST':
        body = json.loads(request.body.decode('utf-8'))
        data = body.get('username')

        # Delete all tokens for the user with the given email
        for token in AuthToken.objects.filter(user__email=data):
            token.delete()

        return JsonResponse('successfully logged out', safe=False)
    return JsonResponse('unsuccessfully logged out', safe=False)


# ---------------- Change User Details ----------------
@api_view(['POST'])
def changeUserDetails(request):
        try:
            body = json.loads(request.body.decode('utf-8'))
            email = body.get('email')

            if not email:
                return Response({'error': 'Email is required.'}, status=status.HTTP_400_BAD_REQUEST)

            try:
                user = CustomUser.objects.get(email=email)
                user.first_name = body.get('firstName', user.first_name)
                user.last_name = body.get('lastName', user.last_name)
                user.save()
                return Response({'message': 'User details updated successfully.'}, status=status.HTTP_200_OK)
            except CustomUser.DoesNotExist:
                return Response({'error': 'User does not exist.'}, status=status.HTTP_404_NOT_FOUND)

        except json.JSONDecodeError:
            return Response({'error': 'Invalid JSON data'}, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
