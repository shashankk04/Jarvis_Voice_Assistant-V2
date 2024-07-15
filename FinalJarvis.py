import logging
import time
from deepgram import (
    SpeakOptions,
    DeepgramClient,
    DeepgramClientOptions,
    LiveTranscriptionEvents,
    LiveOptions,
    Microphone,
)
from groq import Groq
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain

import asyncio
import os
from dotenv import load_dotenv
import pygame

load_dotenv()

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")



class LanguageModelProcessor:
    def __init__(self):
        self.llm = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768", groq_api_key=os.getenv("GROQ_API_KEY"))
        # self.llm = ChatOpenAI(temperature=0, model_name="gpt-4-0125-preview", openai_api_key=os.getenv("OPENAI_API_KEY"))
        # self.llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-0125", openai_api_key=os.getenv("OPENAI_API_KEY"))

        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        # Load the system prompt from a file
        with open('system_prompt.txt', 'r') as file:
            system_prompt = file.read().strip()
        
        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{text}")
        ])

        self.conversation = LLMChain(
            llm=self.llm,
            prompt=self.prompt,
            memory=self.memory
        )

    def process(self, text):
        self.memory.chat_memory.add_user_message(text)  # Add user message to memory

        start_time = time.time()

        # Go get the response from the LLM
        response = self.conversation.invoke({"text": text})
        end_time = time.time()

        self.memory.chat_memory.add_ai_message(response['text'])  # Add AI response to memory

        elapsed_time = int((end_time - start_time) * 1000)
        print(f"LLM ({elapsed_time}ms): {response['text']}")
        return response['text']

class TranscriptCollector:
    def __init__(self):
        self.reset()

    def reset(self):
        self.transcript_parts = []

    def add_part(self, part):
        self.transcript_parts.append(part)

    def get_full_transcript(self):
        return ' '.join(self.transcript_parts)

transcript_collector = TranscriptCollector()

async def get_transcript(handle_full_sentence):
    transcription_complete = asyncio.Event()
    try:
        config = DeepgramClientOptions(options={"keepalive": "true"})
        deepgram: DeepgramClient = DeepgramClient(DEEPGRAM_API_KEY, config)

        dg_connection = deepgram.listen.asynclive.v("1")
        print("Listening...")
        async def on_message(self, result, **kwargs):
            # print (result)
            sentence = result.channel.alternatives[0].transcript
            if not result.speech_final:
                transcript_collector.add_part(sentence)
            else:
                transcript_collector.add_part(sentence)
                full_sentence = transcript_collector.get_full_transcript()
                if len(full_sentence.strip()) > 0:
                    full_sentence = full_sentence.strip()
                    print(f"Human: {full_sentence}")
                    handle_full_sentence(full_sentence)
                    transcript_collector.reset()
                    transcription_complete.set()

        dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)

        options = LiveOptions(
            model="nova-2",
            punctuate=True,
            language="en-US",
            encoding="linear16",
            channels=1,
            sample_rate=16000,
            endpointing=True
        )

        await dg_connection.start(options)
        microphone = Microphone(dg_connection.send)
        microphone.start()
        await transcription_complete.wait()
        microphone.finish()
        await dg_connection.finish()

    except Exception as e:
        print(f"Could not open socket: {e}")
        return

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY)

def send_to_model(messages):
    # response = groq_client.chat.completions.create(
    #     model="llama3-8b-8192",
    #     messages=messages
    #     )
    while True:
        response = groq_client.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=messages
        )
        if response.choices[0].message.content != "":
            break
        
    
    return response
    

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
deepgram_client = DeepgramClient(api_key=DEEPGRAM_API_KEY)

def text_to_speech(text, output_file_path):
    try:
        options = SpeakOptions(
            model="aura-luna-en",  # Change voice if needed
            encoding="linear16",
            container="wav"
        )
        SPEAK_OPTIONS = {"text": text}
        start_time = time.time()
        deepgram_client.speak.v("1").save(output_file_path, SPEAK_OPTIONS, options)
        end_time = time.time()
        elapsed_time = int((end_time - start_time) * 1000)
        print(f"Latency to produce output.wav: ({elapsed_time}ms)")
        play_audio(output_file_path)
    except Exception as e:
        logging.error(f"Failed to convert text to speech: {e}")

def play_audio(file_path):
    try:
        pygame.mixer.init()  # Initialize pygame mixer here
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
        pygame.mixer.music.stop()
        pygame.mixer.quit()  # Quit pygame mixer here
    except pygame.error as e:
        logging.error(f"Failed to play audio: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred while playing audio: {e}")



class ConversationManager:
    def __init__(self):
        self.llm = LanguageModelProcessor()
        self.response=""
    async def main(self):
        def handle_full_sentence(full_sentence):
            self.response = full_sentence
        while True:
            await get_transcript(handle_full_sentence)
            llm_response = self.llm.process(self.response)
            text_to_speech(llm_response, "output.wav")
            self.response=""

if __name__ == "__main__":
    manager = ConversationManager()
    asyncio.run(manager.main())