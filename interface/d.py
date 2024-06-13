from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

@csrf_exempt
def upload_audio(request):
    if request.method == 'POST':
        audio_file = request.FILES['audio']
        gender = request.POST['gender']
        
        # Save the file to the desired location or process it
        with open('path/to/save/' + audio_file.name, 'wb+') as destination:
            for chunk in audio_file.chunks():
                destination.write(chunk)
        
        # Process gender if needed
        # gender is now available in variable 'gender'
        
        return JsonResponse({"status": "success", "message": "Audio uploaded successfully"})
    
    return JsonResponse({"status": "failed", "message": "Invalid request"})


from django.urls import path
from . import views

urlpatterns = [
    path('upload-audio/', views.upload_audio, name='upload_audio'),
]


from django import forms

class AudioUploadForm(forms.Form):
    audio = forms.FileField()
    gender = forms.CharField(max_length=10)


MIDDLEWARE = [
    # Other middleware...
    'django.middleware.csrf.CsrfViewMiddleware',
    # Other middleware...
]


# {% csrf_token %}
