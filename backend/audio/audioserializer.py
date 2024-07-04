from rest_framework import serializers

class audioSerializer(serializers.Serializer):
    audio = serializers.FileField()
