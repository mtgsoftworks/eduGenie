# eduGenie™ - Chatbot Application for Personalized English Language Learning Assistant

eduGenie Chatbot is an interactive, emotionally intelligent learning assistant, fine-tuned specifically for English language learners. Designed to be engaging and empathetic, eduGenie adapts to the user's mood and provides text, audio, and even an animated avatar to enhance the language learning experience.

## Overview

eduGenie combines state-of-the-art language and speech models with real-time animation to create a comprehensive, interactive learning tool. It detects user emotions, generates spoken responses, and presents a dynamic, lifelike avatar, creating a more engaging and supportive environment for language practice.

### Key Features

1. **Emotion Detection**: Leveraging [T5-base emotion detection model](https://huggingface.co/mrm8488/t5-base-finetuned-emotion) from Hugging Face, EduGenie analyzes user inputs and tailors responses to match the detected mood, making interactions feel more personal.

2. **Natural Text-to-Speech (TTS)**: EduGenie’s responses come alive through [SpeechT5 TTS](https://huggingface.co/microsoft/speecht5_tts) and [HiFi-GAN](https://huggingface.co/microsoft/speecht5_hifigan), providing clear and engaging audio output. Fine-tuning on English-specific content ensures quality, making it an ideal companion for learners.

3. **Lifelike Avatar**: Using the **DreamTalk** framework (accessible via [GitHub](https://github.com/ali-vilab/dreamtalk)), EduGenie generates an avatar that lip-syncs with the audio, enhancing user engagement by adding a visual component to the interaction.

4. **Fine-Tuning with Custom Dataset**: The chatbot has been customized with a dedicated English language learning dataset (`dataset_english_fine_tuning.csv`). This fine-tuning was performed on **Google AI Studio** using the **Gemini 1.5 Flash 001 Tuning model**, adapting the chatbot specifically for language learning needs.

5. **Dynamic Response Generation**: EduGenie generates text responses via **Google Gemini API**, using detected emotions to add emotional context to responses—ideal for a conversational learning assistant.

## Quick Start

Here’s how you can set up EduGenie Chatbot on your local machine:

### Step 1: Clone the Repository

```bash
git clone https://github.com/mtgsoftworks/eduGenie.git
cd eduGenie
```

### Step 2: Install Dependencies

- Set up a virtual environment and install required packages:
  ```bash
  python3 -m venv env
  source env/bin/activate
  pip install -r requirements.txt
  ```
- Drag the 'checkpoints' folder you downloaded from this link https://drive.google.com/file/d/1BFg7pFMS5DNsNDtHWe2eLINpxJbAua59/view?usp=sharing directly to the dreamtalk folder.
 
- Install the Desktop development with C++ module and install the CMake software build automation program. (Don't forget to set the environmental variables path for CCmake)  
  

### Step 3: Configure the Models

- Download models for emotion detection, text-to-speech, and vocoder:
  - Emotion Detection: `t5-base-finetuned-emotion`
  - TTS: `speecht5_tts` and `speecht5_hifigan`

### Step 4: Set Up API Key

- Replace `'YOUR_API_KEY'` in the code with your **Google Gemini API** key to enable response generation.

## Project Components

- **app.py**: Main Flask application for chatbot interactions and video file serving.
- **video_generate.py**: Generates avatar video synchronized with the chatbot’s audio output using DreamTalk.
- **text_to_speech.py**: Converts chatbot responses to audio using SpeechT5.
- **detect_emotion.py**: Identifies emotions from user inputs, shaping responses accordingly.
- **generate_response.py**: Uses Google Gemini API to create contextually relevant responses, incorporating emotional insights from `detect_emotion.py`.

## Usage

1. **Run the Application**:
   ```bash
   python app.py
   ```

2. **Access the Chat Interface**:
   - Open your browser and navigate to `http://localhost:5000`.

3. **Start Chatting with EduGenie**:
   - Enter questions or statements in the chat box, and EduGenie will respond with text and an animated video!

## Dependencies

eduGenie Chatbot relies on several key libraries and models:
- **Transformers** for language and emotion models
- **MoviePy** for video processing
- **Flask** for creating a local web server
- **Google Gemini API** for text generation and response adaptation

---

eduGenie Chatbot is more than just a language learning tool—it's a responsive, engaging, and empathetic virtual assistant designed to make learning English interactive and enjoyable!
