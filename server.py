import asyncio
import websockets
import json
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
import random

# video pipeline

# other dependencies
import time

"""
### Document ###

# run command
RealtimeSpeech/real-time-speech/server.py -> python server.py
Motion/PoseSample -> npm run dev

# check
Motion/PythonSample/server.py -> Replay sound + websocket
RealtimeSpeech/real-time-speech/texttospeech.py -> create tts file and play video pipeline and audio pipeline at the same time

# ToDo:
websocket insufficient resource error
inaccuracy of speech recognition
inaccuracy of lipsync with wav file -> shape analysis + websocket failed, random sampling

"""




"""
# pyaudio install
pip install --global-option='build_ext' --global-option='-I/usr/local/include' --global-option='-L/usr/local/lib' pyaudio
"""

# Audio recording parameters
RATE = 16000
CHUNK = int(RATE / 10)  # 100ms


# OpenAI Part
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

# [finish openai part]

# Video Pipeline

# Audio pipeline

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

notes = ["A", "I", "U", "E", "O"]


async def audio(websocket, path):
    rounds = 1
    chat_log = None

    while True:
        try:
            print('streaming loop :' + str(rounds))

            rp = await websocket.recv()
            
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

                        data = {
                            "open": "open"
                        }
                        data = json.dumps(data)
                        await websocket.send(data)

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
                            d = json.dumps({
                                "open": "open",
                                "note": random.choice(notes)
                            })
                            await websocket.send(d)
                            
                        stream.close()
                        p.terminate()

                        data = {
                            "open": "close"
                        }
                        data = json.dumps(data)

                        await websocket.send(data)

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


"""
async def echo(websocket, path):

    rp = await websocket.recv()
    
    time.sleep(6)

    filename = "out_test.wav"

    data = "open"

    await websocket.send(data)

    try:
        wf = wave.open(filename, "rb")
    except FileNotFoundError: #ファイルが存在しなかった場合
        print("[Error 404] No such file or directory: " + Filename)
                                
    # create stream
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

    data = "close"

    
    await websocket.send(data)
"""

async def main():
    
    async with websockets.serve(audio, "localhost", 8765):
        await asyncio.Future()



asyncio.run(main())