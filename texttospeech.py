"""
# audio pipeline
from google.cloud import speech_v1 as speech
from google.cloud.speech_v1 import enums
from google.cloud.speech_v1 import types
import pyaudio
from google.cloud import texttospeech
import subprocess
from six.moves import queue
import openai
import wave

# video pipeline
import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=True,
        refine_face_landmarks=True)

import tensorflow as tf
from tf_bodypix.api import download_model, load_model, BodyPixModelPaths

import threading


# instantiate bodypix
bodypix_model = load_model(download_model(BodyPixModelPaths.MOBILENET_FLOAT_50_STRIDE_16))
cap = cv2.VideoCapture(0)
_, init_f = cap.read()
width, height, _ = init_f.shape
bg_img = cv2.resize(cv2.imread("bg.jpeg"), (height, width))



APIKEY = "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"
openai.api_key = APIKEY
completion = openai.Completion()

start_chat_log = '''
私：こんにちは、調子はいかがですか？
AI：非常に良いですよ。今日は何をお手伝いしましょうか？
'''

def ask(question, chat_log=None):
    if chat_log is None:
        chat_log = start_chat_log
    prompt = f'{chat_log}私: {question}\nAI:'
    response = completion.create(
        prompt=prompt, engine="davinci", stop=['\n'], temperature=0.9,
        top_p=1, frequency_penalty=0, presence_penalty=0.6, best_of=1,
        max_tokens=150)
    answer = response.choices[0].text.strip()
    return answer

def append_interaction_to_chat_log(question, answer, chat_log=None):
    if chat_log is None:
        chat_log = start_chat_log
    return f'{chat_log}私: {question}\nAI: {answer}\n'




# Audio recording parameters
RATE = 16000
CHUNK = int(RATE / 10)  # 100ms


class MicrophoneStream(object):
    
    
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

def video_pipeline():

    threading.Thread(target=audio_pipeline).start()

    # video pipeline settings

    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()

        if ret:
            
            result = bodypix_model.predict_single(frame)
            mask = result.get_mask(threshold=0.5).numpy().astype(np.uint8)
            masked_image = cv2.bitwise_and(frame, frame, mask=mask)

            neg = np.add(mask, -1)
            inverse = np.where(neg==-1, 1, neg).astype(np.uint8)
            masked_bg = cv2.bitwise_and(bg_img, bg_img, mask=inverse)
            final = cv2.add(masked_image, masked_bg)

            frame = final
            

            # cv2.imshow("Demo", frame)

            
            frame.flags.writeable = False
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = holistic.process(frame)
            
            frame.flags.writeable = True
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            mp_drawing.draw_landmarks(
                frame,
                results.face_landmarks,
                mp_holistic.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_contours_style())
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles
                .get_default_pose_landmarks_style())

            
            cv2.imshow("Demo", cv2.flip(frame, 1))
            

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    



def audio_pipeline():

    language_code = 'ja-JP'  # a BCP-47 language tag

    speechclient = speech.SpeechClient()
    config = types.RecognitionConfig(
            encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=RATE,
            language_code=language_code)
    streaming_config = types.StreamingRecognitionConfig(
            config=config,
            interim_results=True)

    textspeechclient = texttospeech.TextToSpeechClient()

    voice = texttospeech.VoiceSelectionParams(
        language_code="ja-JP", ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
    )

    # Select the type of audio file you want returned
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )


    rounds = 1
    chat_log = None




    while True:
        try:
            print('streaming loop :' + str(rounds))
            with MicrophoneStream(RATE, CHUNK) as stream:
                audio_generator = stream.generator()
                # Create request data
                requests = (types.StreamingRecognizeRequest(audio_content=content) for content in audio_generator)
                # POST data to google cloud speech
                responses = speechclient.streaming_recognize(streaming_config, requests)

                num_chars_printed = 0
                for response in responses:
                    if not response.results:
                        continue

                    result = response.results[0]

                    if not result.alternatives:
                        continue

                    transcript = result.alternatives[0].transcript
                    # print('Microphone input: ' + transcript)

                    overwrite_chars = ' ' * (num_chars_printed - len(transcript))

                    # print(result.stability)
                    if result.stability >= 0.899:
                        question = transcript + overwrite_chars

                        print("question:", question)
                        answer = ask(question)


                        synthesis_input = texttospeech.SynthesisInput(text=answer)
                        
                        tsresponse = textspeechclient.synthesize_speech(
                            input=synthesis_input, voice=voice, audio_config=audio_config
                        )

                        # The response's audio_content is binary.

                        with open("output.wav", "wb") as out:
                            # Write the response to the output file.
                            out.write(tsresponse.audio_content)
                            print('Audio content written to file "output.wav"')

                        subprocess.call("ffmpeg -i {} out_test.wav -y".format("output.wav"), shell=True)

                        Filename = "out_test.wav"

                        try:
                            wf = wave.open(Filename, "rb")
                        except FileNotFoundError: #ファイルが存在しなかった場合
                            print("[Error 404] No such file or directory: " + Filename)
                                
                        # ストリームを開く
                        p = pyaudio.PyAudio()
                        stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                                        channels=wf.getnchannels(),
                                        rate=wf.getframerate(),
                                        output=True)

                        # 音声を再生
                        chunk = 1024
                        data = wf.readframes(chunk)
                        while data != b'':
                            stream.write(data)
                            data = wf.readframes(chunk)
                            
                        stream.close()
                        p.terminate()


                        print("answer:", answer)
                        chat_log = append_interaction_to_chat_log(responses, answer)

                        

                        
                        # text_translation(responses)
                        MicrophoneStream(RATE, CHUNK)._clear_buffer()
                    else:
                        # sys.stdout.write(transcript + overwrite_chars + '\r')
                        # sys.stdout.flush()
                        
                        num_chars_printed = len(transcript)


                

                
        except Exception as err:
            print(err)
            rounds += 1

def video_mediapipe():
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=True,
        refine_face_landmarks=True)

    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)

        # Draw landmark annotation on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(
            image,
            results.face_landmarks,
            mp_holistic.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_contours_style())
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles
            .get_default_pose_landmarks_style())
        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('MediaPipe Holistic', cv2.flip(image, 1))
        if cv2.waitKey(5) & 0xFF == 27:
            break
    cap.release()


if __name__ == "__main__":
    # audio_pipeline()
    video_pipeline()
    # video_mediapipe()
"""



import argparse
from google.cloud import texttospeech
import wave
import subprocess
import pyaudio

parser = argparse.ArgumentParser()
parser.add_argument("--text", type=str, 
default="今日は水曜日です。とても天気が良くてお散歩したくなります。")

args = parser.parse_args()

# Instantiates a client
client = texttospeech.TextToSpeechClient()

# Set the text input to be synthesized
synthesis_input = texttospeech.SynthesisInput(text=args.text)

# Build the voice request, select the language code ("en-US") and the ssml
# voice gender ("neutral")
voice = texttospeech.VoiceSelectionParams(
    language_code="ja-JP", ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
)

# Select the type of audio file you want returned
audio_config = texttospeech.AudioConfig(
    audio_encoding=texttospeech.AudioEncoding.MP3
)

# Perform the text-to-speech request on the text input with the selected
# voice parameters and audio file type
response = client.synthesize_speech(
    input=synthesis_input, voice=voice, audio_config=audio_config
)

# The response's audio_content is binary.

with open("output.wav", "wb") as out:
    # Write the response to the output file.
    out.write(response.audio_content)
    print('Audio content written to file "output.mp3"')

subprocess.call("ffmpeg -i {} out_test.wav -y".format("output.wav"), shell=True)

Filename = "out_test.wav"

try:
    wf = wave.open(Filename, "rb")
except FileNotFoundError: #ファイルが存在しなかった場合
    print("[Error 404] No such file or directory: " + Filename)
        
# ストリームを開く
p = pyaudio.PyAudio()
stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(),
                rate=wf.getframerate(),
                output=True)

# 音声を再生
chunk = 1024
data = wf.readframes(chunk)
while data != b'':
    stream.write(data)
    data = wf.readframes(chunk)
stream.close()
p.terminate()
