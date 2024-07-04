from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from .audioserializer import audioSerializer

from keras.models import load_model # type: ignore
import numpy as np
import librosa

# Load the saved model
model = load_model(r'C:\Users\Jayanth\Documents\GitHub\jubilant-funicular\backend\assets\audio\GenderModelCNN.hdf5')

def extract_features(file_path):
    audio, sample_rate = librosa.load(file_path)
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
    return mfccs_scaled_features

class AudioUploadView(APIView):
    def post(self, request, *args, **kwargs):
        serializer = audioSerializer(data=request.data)
        if serializer.is_valid():
            audio_file = serializer.validated_data['audio']
            path = default_storage.save('audio/' + audio_file.name, ContentFile(audio_file.read()))

            # Extract features from the uploaded audio file
            file_path = default_storage.path(path)
            features = extract_features(file_path)
            features = features.reshape((1, features.shape[0], 1))

            # Make prediction
            prediction = model.predict(features)
            predicted_class = np.argmax(prediction, axis=1)[0]

            return Response({
                "message": "File uploaded and processed successfully!",
                "path": path,
                "prediction": int(predicted_class)
            }, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
