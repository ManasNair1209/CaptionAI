import os
import cv2
import torch
import gradio as gr
import pandas as pd
from PIL import Image
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer, pipeline
from moviepy.editor import ImageSequenceClip
from gtts import gTTS
import re

# Load pre-trained models for image captioning
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Generation settings for captions
max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

# Function to extract frames from the video
def extract_frames(video_path, output_folder="frames"):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("Error opening video file")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    frame_paths = []
    frame_number = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(output_folder, f"frame_{frame_number:04d}.png")
        cv2.imwrite(frame_path, frame)
        frame_paths.append(frame_path)
        frame_number += 1

    cap.release()
    return frame_paths

# Function to generate captions for all frames
def generate_captions(image_paths):
    captions = []
    for img_path in image_paths:
        i_image = Image.open(img_path).convert("RGB")
        pixel_values = feature_extractor(images=[i_image], return_tensors="pt").pixel_values.to(device)
        output_ids = model.generate(pixel_values, **gen_kwargs)
        caption = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        captions.append(caption)
    return captions

# Function to overlay captions on frames
def create_captioned_video(frame_paths, captions, output_video_path, fps=24):
    frame_width, frame_height = 640, 480  # Adjust depending on your frame size
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    for frame_path, caption in zip(frame_paths, captions):
        img = cv2.imread(frame_path)
        img = cv2.resize(img, (frame_width, frame_height))

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_thickness = 2
        text_size = cv2.getTextSize(caption, font, font_scale, font_thickness)[0]
        text_x = (img.shape[1] - text_size[0]) // 2
        text_y = img.shape[0] - 50

        cv2.rectangle(img, (text_x - 10, text_y - text_size[1] - 10),
                      (text_x + text_size[0] + 10, text_y + 10), (0, 0, 0), -1)
        cv2.putText(img, caption, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness, lineType=cv2.LINE_AA)
        video_writer.write(img)

    video_writer.release()

# Function to clean and summarize the generated captions
def clean_caption(caption):
    caption = re.sub(r'\b(\w+\s+)\1{1,}', r'\1', caption)  # Remove duplicate words
    caption = re.sub(r'\s+', ' ', caption).strip()  # Remove extra whitespace
    return caption

def chunk_text(text, max_length=1000):
    words = text.split(' ')
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        if current_length + len(word) + 1 > max_length:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_length = 0
        current_chunk.append(word)
        current_length += len(word) + 1

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

def summarize_chunk(chunk, summarizer, max_length=100, min_length=50):
    summary = summarizer(chunk, max_length=max_length, min_length=min_length, do_sample=False)
    return summary[0]['summary_text']

# Improved summarization function using t5-large model
def generate_summary_from_captions(captions):
    combined_text = ' '.join([clean_caption(c) for c in captions])
    text_chunks = chunk_text(combined_text, max_length=800)  # Reduced chunk size

    # Load summarization model (t5-large)
    summarizer = pipeline("summarization", model="t5-large")

    summarized_chunks = [summarize_chunk(chunk, summarizer) for chunk in text_chunks]
    final_summary = ' '.join(summarized_chunks)

    # Post-process to remove redundancy
    final_summary = clean_caption(final_summary)
    return final_summary

# Function to convert text to speech
def text_to_speech(text, output_file="output.mp3", lang="en", slow=False):
    tts = gTTS(text=text, lang=lang, slow=slow)
    tts.save(output_file)
    return output_file

# Main function for Gradio
def process_video(video):
    # Extract frames from the video
    frame_paths = extract_frames(video)

    # Generate captions for each frame
    captions = generate_captions(frame_paths)

    # Create captioned video
    output_video = "output_captioned_video.mp4"
    create_captioned_video(frame_paths, captions, output_video)

    # Generate video summary
    summary = generate_summary_from_captions(captions)

    # Convert summary to speech
    audio_file = text_to_speech(summary, output_file="summary_audio.mp3")

    return output_video, summary, audio_file

# Gradio interface
interface = gr.Interface(
    fn=process_video,
    inputs=gr.Video(label="Upload Video"),
    outputs=[
        gr.File(label="Download Captioned Video"),  # Captioned video output
        gr.Textbox(label="Video Summary"),  # Display text summary
        gr.Audio(label="Listen to Summary")  # Audio of the summary
    ],
    title="CaptionAI",
    description="Upload a video, generate captions for each frame, see a summarized text, and listen to the summary as audio."
)

# Launch the interface
interface.launch(debug=True)
