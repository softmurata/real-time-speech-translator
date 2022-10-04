import pyaudio
import cv2
import multiprocessing
from multiprocessing import Process
import numpy as np
from google.cloud import translate_v2 as translate
from google.cloud import speech_v1 as speech
from google.cloud.speech_v1 import enums
from google.cloud.speech_v1 import types

translate_client = translate.Client()

CHUNK = 1024
CHANNELS = 1
RATE = 44100
FORMAT = pyaudio.paInt16
THRESHOLD = 0.02

language_code = 'ja-JP'  # a BCP-47 language tag

client = speech.SpeechClient()
config = types.RecognitionConfig(
        encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code=language_code)
streaming_config = types.StreamingRecognitionConfig(
        config=config,
        interim_results=True)

def input_voice():
    setting = 0

    p = pyaudio.PyAudio()
    
    stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                output=True,
                frames_per_buffer=CHUNK)

    while True:
        data = stream.read(CHUNK)
        x = np.frombuffer(data, dtype="int16") / 32768

        if x.max() > THRESHOLD:
            all = []
            all.append(data)
            for i in range(0, int(RATE / CHUNK * 4)):
                data = stream.read(CHUNK)
                all.append(data)
            data = b''.join(all)
            # Create request data
            # requests = (types.StreamingRecognizeRequest(audio_content=content) for content in audio_generator)
            # POST data to google cloud speech
            # responses = client.streaming_recognize(streaming_config, requests)
            stream.write(data, CHUNK)
    
    stream.stop_stream()
    stream.close()
    p.terminate()


def main():
    
    voice = Process(target=input_voice)
    voice.start()

    cap = cv2.VideoCapture(0)

    img_last = None

    while voice.is_alive() == True:
        ret, frame = cap.read()

        cv2.imshow("Camera", frame)

        if cv2.waitKey(1) == 13:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()