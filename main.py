from google.cloud import translate_v2 as translate
from google.cloud import speech_v1 as speech
from google.cloud.speech_v1 import enums
from google.cloud.speech_v1 import types
import pyaudio
from six.moves import queue
import sys
import openai

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

RATE = 16000
CHUNK = int(RATE)  # 100ms

language_code = 'ja-JP'  # a BCP-47 language tag
client = speech.SpeechClient()

config = types.RecognitionConfig(
            encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=RATE,
            language_code=language_code)
streaming_config = types.StreamingRecognitionConfig(
            config=config,
            interim_results=True)


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
            responses = client.streaming_recognize(streaming_config, requests)

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



