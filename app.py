import subprocess
from pathlib import Path
from moviepy.editor import VideoFileClip, AudioFileClip
import google.generativeai as genai
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from flask import Flask, render_template, request, jsonify
import logging
import os
import time
from datetime import datetime
from uuid import uuid4
from datasets import load_dataset
import torch
import soundfile as sf
from flask import send_from_directory

app = Flask(__name__)

os.makedirs('logs', exist_ok=True)

local_cache_dir_emotions = "./models/t5-base-finetuned-emotion/"
local_cache_dir_tts = "./models/speecht5_tts/"
local_cache_dir_dataset = './models/cmu-arctic-xvector/'

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_filename = f'logs/chatbot_interactions_{timestamp}.log'

logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

local_username = "admin"

genai.configure(api_key='AIzaSyA9b4YNiAsDC7qCIj05F822GqxVSaTwj8Y')  # Replace with your actual API key

gemini_API = genai.GenerativeModel('gemini-1.5-pro')

tokenizer = AutoTokenizer.from_pretrained(
    "mrm8488/t5-base-finetuned-emotion",
    cache_dir=local_cache_dir_emotions,
    force_download=False
)
emotion_model = AutoModelForSeq2SeqLM.from_pretrained(
    "mrm8488/t5-base-finetuned-emotion",
    cache_dir=local_cache_dir_emotions,
    force_download=False
)

processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts", cache_dir=local_cache_dir_tts)
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts", cache_dir=local_cache_dir_tts)
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan", cache_dir=local_cache_dir_tts)

embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation", cache_dir=local_cache_dir_dataset)
speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

current_directory = os.getcwd()


def video_generate(audio_name, video_name):
    print("Avatar Talking Head Video Generate is Starting...")
    final_directory = os.path.join(os.getcwd(), 'dreamtalk')
    os.chdir(final_directory)

    command = [
        'python', 'main.py',
        '--wav_path', os.path.join(current_directory, 'static', audio_name),
        '--style_clip_path', os.path.join(current_directory, 'dreamtalk', 'data', 'style_clip', '3DMM', 'M030_front_happy_level3_001.mat'),
        '--pose_path', os.path.join(current_directory, 'dreamtalk', 'data', 'pose', 'RichardShelby_front_neutral_level1_001.mat'),
        '--image_path', os.path.join(current_directory, 'dreamtalk', 'data', 'src_img', 'avatar', 'avatar.jpg'),
        '--cfg_scale', '1.0',
        '--max_gen_len', '60',
        '--output_name', video_name,
    ]

    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        start_time = time.time()

        while True:
            time.sleep(1)
            if process.poll() is not None:
                break
            if time.time() - start_time > 60:
                print("Process is taking too long. Terminating...")
                process.terminate()
                break

        stdout, stderr = process.communicate()
        print("Output:", stdout.decode())
        os.chdir('..')
        if stderr:
            print("Errors:", stderr.decode())
            process.terminate()

    except Exception as e:
        print("An error occurred:", e)
        process.terminate()



def combine_audio_video(video_path, audio_path, output_path):
    print("Audio and Video Being Combined...")
    video = VideoFileClip(os.path.join(current_directory, video_path))
    audio = AudioFileClip(os.path.join(current_directory, audio_path))

    video = video.set_audio(audio)
    video.write_videofile(os.path.join(current_directory, output_path), codec='libx264', audio_codec='aac')


def generate_response(prompt):
    response = gemini_API.generate_content(prompt)
    print("Response Successfully Created")
    return response.text

def text_to_speech(user, input_text):
    logging.basicConfig(level=logging.INFO)
    message_id = str(uuid4())
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{user}_{timestamp}_{message_id}.mp3"
    audio_path = os.path.join(current_directory, 'static', filename)

    max_length = 600
    inputs = processor(text=input_text, return_tensors="pt", max_length=max_length, truncation=True)

    try:
        logging.info("Generating speech...")
        print("Generating speech...")
        speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
        
        if speech is not None and len(speech.numpy()) > 0:
            sf.write(audio_path, speech.numpy(), samplerate=16000)
            logging.info(f"Audio file '{audio_path}' successfully created.")
            print(f"Audio file '{audio_path}' successfully created.")
            return f"/static/{filename}"
        else:
            logging.error("Generated speech is empty or None.")
            print("Generated speech is empty or None.")
            return None
    except Exception as e:
        logging.error(f"An error occurred in text_to_speech: {e}")
        print(f"An error occurred in text_to_speech: {e}")
        return None



def detect_emotion(user_input):
    input_text = "classify: " + user_input
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    output = emotion_model.generate(input_ids)
    emotion = tokenizer.decode(output[0], skip_special_tokens=True)
    print("Emotion Detected: ", emotion)
    return emotion

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    user_input = request.form['user_input']
    detected_emotion = detect_emotion(user_input)
    prompt = (
        f"User says: '{user_input}'. "
        f"Emotion detected: '{detected_emotion}'. "
        f"As a helpful learning companion, provide an empathetic and engaging response "
        f"that addresses the user's emotional state. "
        f"Also, consider asking a follow-up question to encourage further interaction. "
        f"Make sure to be supportive and offer relevant learning resources or tips."
    )
    chatbot_response = generate_response(prompt)

    audio_path = text_to_speech(local_username, chatbot_response)
    video_name_local = str(uuid4())
    audio_name = os.path.basename(audio_path)

    logging.info(f"User input received: {user_input} | Detected emotion: {detected_emotion}")
    logging.info(f"Chatbot response generated: {chatbot_response}")

    try:

        video_generate(audio_name, video_name_local)
        input_video_path = os.path.join(current_directory, 'dreamtalk', 'output_video', video_name_local + '.mp4')
        output_path = os.path.join(current_directory, "output", video_name_local + '.mp4')
        combine_audio_video(input_video_path, os.path.join(current_directory, 'static', audio_name), output_path)

        logging.info(f"User: {user_input} | Emotion: {detected_emotion} | Bot: {chatbot_response}")

        logging.info(f"Generated video at: {output_path}")

    except Exception as e:    
        logging.error(f"Error in video generation: {e}")

    return jsonify({
        'response': chatbot_response,
        'emotion': detected_emotion,
        "audio_path": audio_path,
        "video_path": video_name_local + '.mp4'
    })


@app.route('/feedback', methods=['POST'])
def feedback():
    user_feedback = request.form['user_feedback']
    user_input = request.form['user_input']
    logging.info(f"Feedback received: {user_feedback} for User input: {user_input}")
    return jsonify({'status': 'Feedback logged'})

@app.route('/output/<path:filename>')
def serve_video(filename):
    return send_from_directory("output", filename)


if __name__ == '__main__':
    app.run(host='localhost', port=5000, debug=True, threaded=True)
