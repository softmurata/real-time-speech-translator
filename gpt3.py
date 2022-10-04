import openai
from dotenv import load_dotenv

"""
https://www.twilio.com/blog/openai-gpt-3-chatbot-python-twilio-sms
"""

load_dotenv()

APIKEY = env.API_KEY
openai.api_key = APIKEY

completion = openai.Completion()

start_chat_log = '''Human: Hello, who are you?
AI: I am doing great. How can I help you today?
'''

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


chat_log = None
question = '今日は何曜日ですか？'
answer = ask(question, chat_log)
print(answer)
question = 'いい天気でしたか？'
chat_log = append_interaction_to_chat_log(question, answer, chat_log)
answer = ask(question)
print(answer)


