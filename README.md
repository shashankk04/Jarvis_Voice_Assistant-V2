# Jarvis Voice Assistant

Jarvis Voice Assistant is an intelligent voice assistant leveraging state-of-the-art language models and speech recognition technologies. It is designed to listen to real-time audio input, transcribe it, process the text using a sophisticated language model, and respond with generated speech. This assistant facilitates seamless human-computer interactions through natural language processing and speech synthesis.

## Features

- **Real-Time Speech Recognition**: Utilizes Deepgram's API for accurate and fast transcription of spoken language.
- **Language Model Integration**: Employs Groq's powerful language models to generate contextually relevant responses.
- **Text-to-Speech Conversion**: Converts text responses into natural-sounding speech using Deepgram's text-to-speech capabilities.
- **Memory Retention**: Maintains conversation history to provide coherent and context-aware interactions.
- **Customizable System Prompts**: Allows loading system prompts from external files to tailor the assistant's behavior.

## Prerequisites

- Python 3.7 or later
- Deepgram API Key
- Groq API Key

## Components

### LanguageModelProcessor

Handles interaction with the Groq language model, processes user inputs, and generates responses while maintaining conversation history.

### TranscriptCollector

Collects and manages parts of the transcript received from the speech recognition module.

### ConversationManager

Coordinates the main workflow of the assistant, including receiving transcripts, processing them through the language model, and generating spoken responses.

### get_transcript

An asynchronous function that handles live transcription using Deepgram's API and processes complete sentences through the language model.

### text_to_speech

Converts text responses from the language model into audio files using Deepgram's text-to-speech capabilities and plays the generated audio.
