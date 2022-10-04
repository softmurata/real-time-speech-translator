import pyaudio
import wave
import numpy as np
from datetime import datetime
import os
import multiprocessing
from multiprocessing import Process
import cv2
from google.cloud import speech_v1 as speech
from google.cloud.speech_v1 import enums
from google.cloud.speech_v1 import types
from six.moves import queue

cv2_base_dir = os.path.dirname(os.path.abspath(cv2.__file__))
haar_model = os.path.join(cv2_base_dir, 'data/haarcascade_frontalface_default.xml')
cascade = cv2.CascadeClassifier(haar_model)

# 音データフォーマット
FORMAT = pyaudio.paInt16
CHANNELS = 1
threshold = 0.1
RATE = 16000
chunk = int(RATE / 10)  # 100ms



class MicrophoneStream(object):
    """Opens a recording stream as a generator yielding the audio chunks."""
    
    def __init__(self, rate, chunk):
        
        self._rate = rate
        self._chunk = chunk

        # Create a thread-safe buffer of audio data
        self._buff = queue.Queue()
        self.closed = True

    def __enter__(self):
        
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            # The API currently only supports 1-channel (mono) audio
            # https://goo.gl/z757pE
            channels=1, rate=self._rate,
            input=True, frames_per_buffer=self._chunk,
            # Run the audio stream asynchronously to fill the buffer object.
            # This is necessary so that the input device's buffer doesn't
            # overflow while the calling thread makes network requests, etc.
            stream_callback=self._fill_buffer,
        )

        self.closed = False

        return self

    def __exit__(self, type, value, traceback):
        
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed = True
        # Signal the generator to terminate so that the client's
        # streaming_recognize method will not block the process termination.
        self._buff.put(None)
        self._audio_interface.terminate()

    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        """Continuously collect data from the audio stream, into the buffer."""
        
        self._buff.put(in_data)
        return None, pyaudio.paContinue
    
    def _clear_buffer(self):
        
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        # self.closed = True
        # Signal the generator to terminate so that the client's
        # streaming_recognize method will not block the process termination.
        self._buff.put(None)
        self._audio_interface.terminate()

    def generator(self):

        while not self.closed:
            # Use a blocking get() to ensure there's at least one chunk of
            # data, and stop iteration if the chunk is None, indicating the
            # end of the audio stream.
            chunk = self._buff.get()
            if chunk is None:
                return
            data = [chunk]

            # Now consume whatever other data's still buffered.
            while True:
                try:
                    chunk = self._buff.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk)
                except queue.Empty:
                    break

            yield b''.join(data)


language_code = 'ja-JP'  # a BCP-47 language tag

client = speech.SpeechClient()
config = types.RecognitionConfig(
        encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code=language_code)
streaming_config = types.StreamingRecognitionConfig(
        config=config,
        interim_results=True)

# 音声処理部
def input_voice():
    rounds = 1
    while True:
        try:
            print('streaming loop :' + str(rounds))
            with MicrophoneStream(RATE, chunk) as stream:
                audio_generator = stream.generator()
                # Create request data
                requests = (types.StreamingRecognizeRequest(audio_content=content) for content in audio_generator)
                # POST data to google cloud speech
                responses = client.streaming_recognize(streaming_config, requests)

                for response in responses:
                    if not response.results:
                        continue

                    result = response.results[0]
                    if not result.alternatives:
                        continue
                    
                    # Display the transcription of the top alternative.
                    transcript = result.alternatives[0].transcript
                    print('Microphone input: ' + transcript)
                    in_text = transcript

                    fileobj = open("./test.txt", "a", encoding = "utf-8")
                    fileobj.write(in_text)
                    fileobj.close()

        except Exception as err:
            print(err)
            rounds += 1

def main():
    voice = Process(target=input_voice)
    voice.start()
    cap = cv2.VideoCapture(0)

    img_last = None

    # voice()処理が完了(＝False)するまで継続
    while voice.is_alive() == True:
        _, frame = cap.read()
        frame = cv2.resize(frame, (800,400))
        gray0 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray0, (9,9), 0)
        img_b = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)[1]

        if img_last is None:
            img_last = img_b
            continue
        frame_diff = cv2.absdiff(img_last, img_b)
        cnts = cv2.findContours(frame_diff, cv2.RETR_EXTERNAL, 
                                cv2.CHAIN_APPROX_SIMPLE)[0]

        face_listA = cascade.detectMultiScale(gray0, minSize = (150, 150))

        for (x,y,w,h) in face_listA:
            img = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)#顔をキャプチャ

        img_last = img_b
        cv2.imshow("Diff Camera", frame)

        # このコードを消すとなぜかフリーズする
        if cv2.waitKey(1)==13: break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
