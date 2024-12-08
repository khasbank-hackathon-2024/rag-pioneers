import os
from google.cloud import texttospeech as tts
import base64
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '../key/owner.json'
client = tts.TextToSpeechClient()

voice = tts.VoiceSelectionParams(
    language_code = "ru-Ru",
    name = "ru-RU-Wavenet-B"
)
config = tts.AudioConfig(
    audio_encoding = tts.AudioEncoding.MP3,
    speaking_rate = 1,
    pitch=1
)

def ttsservice(text: str):
    synthesis_input = tts.SynthesisInput(text = text)
    response = client.synthesize_speech(
        input = synthesis_input,
        voice = voice,
        audio_config =config
    )
    return base64.b64encode(response.audio_content).decode("utf-8")
