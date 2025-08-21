import streamlit as st
import torch
from diffusers import AutoPipelineForText2Image
from gtts import gTTS
from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips
import re
import gc
import google.generativeai as genai
import os
import requests
import base64

# --- Page Configuration ---
st.set_page_config(
    page_title="VidScribe AI",
    page_icon="🎬",
    layout="centered"
)

st.title("🎬 VidScribe AI")
st.markdown("Turn any topic into a short video with AI-generated narration and images.")

# --- API Key Management (In-App Prompt Only) ---
st.sidebar.header("🔑 API Configuration")
st.session_state.gemini_key = st.sidebar.text_input(
    label="Enter your Gemini API Key", type="password", placeholder="Paste Gemini key here..."
)
st.session_state.stability_key = st.sidebar.text_input(
    label="Enter your Stability AI Key", type="password", placeholder="Paste Stability AI key here..."
)
st.session_state.hf_token = st.sidebar.text_input(
    label="Enter your Hugging Face Token", type="password", placeholder="Paste Hugging Face token here..."
)

# --- Model & Helper Functions ---

# Text Generation
def generate_story_with_gemini(topic):
    genai.configure(api_key=st.session_state.gemini_key)
    model = genai.GenerativeModel('gemini-1.5-pro-latest')
    prompt = f"You are a master storyteller. Write a captivating short story about '{topic}'. The story should have a clear narrative arc and be approximately 3-4 paragraphs long."
    response = model.generate_content(prompt)
    return response.text

def generate_story_with_hf(topic):
    API_URL = "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B-Instruct"
    headers = {"Authorization": f"Bearer {st.session_state.hf_token}"}
    prompt = f"You are a master storyteller. Write a captivating short story about '{topic}'. The story should have a clear narrative arc and be approximately 3-4 paragraphs long."
    response = requests.post(API_URL, headers=headers, json={"inputs": prompt, "parameters": {"max_new_tokens": 512, "return_full_text": False}})
    if response.status_code != 200:
        raise ConnectionError(f"Hugging Face API Error: {response.text}")
    return response.json()[0]['generated_text']

# Image Generation
@st.cache_resource(show_spinner="Loading local image models...")
def load_diffusion_pipelines():
    base = AutoPipelineForText2Image.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safensors=True).to("cuda")
    refiner = AutoPipelineForText2Image.from_pretrained("stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, use_safensors=True, variant="fp16").to("cuda")
    return base, refiner

def generate_images_local(image_prompts):
    base, refiner = load_diffusion_pipelines()
    image_files = []
    # ... (rest of local generation logic)
    return image_files
    
def generate_images_api(image_prompts):
    image_files = []
    for i, prompt in enumerate(image_prompts):
        styled_prompt = f"{prompt}, cinematic, masterpiece, 8k, high detail"
        response = requests.post(
            "https://api.stability.ai/v1/generation/stable-diffusion-v1-6/text-to-image",
            headers={"Authorization": f"Bearer {st.session_state.stability_key}", "Accept": "application/json"},
            json={"text_prompts": [{"text": styled_prompt}], "samples": 1, "steps": 30}
        )
        if response.status_code != 200:
            raise ConnectionError(f"Stability AI API Error: {response.text}")
        data = response.json()
        filename = f"generated_image_{i}.png"
        with open(filename, "wb") as f:
            f.write(base64.b64decode(data["artifacts"][0]["base64"]))
        image_files.append(filename)
    return image_files

# Video Assembly
def create_video(story_text, image_files):
    # ... (video assembly logic remains the same)
    tts = gTTS(story_text, lang='en')
    narration_file = "narration.mp3"
    tts.save(narration_file)
    audioclip = AudioFileClip(narration_file)
    duration_per_image = audioclip.duration / len(image_files)
    clips = [ImageClip(img_file).set_duration(duration_per_image) for img_file in image_files]
    slideshow = concatenate_videoclips(clips, method="compose")
    final_video = slideshow.set_audio(audioclip)
    video_filename = "final_video.mp4"
    final_video.write_videofile(video_filename, fps=24, codec='libx264')
    return video_filename, narration_file


# --- Main App Interface ---
st.sidebar.header("⚙️ Model Selection")
text_model_option = st.sidebar.selectbox("Choose a Storyteller:", ("Google Gemini", "Hugging Face (Llama 3)"))
image_model_option = st.sidebar.selectbox("Choose an Image Artist:", ("Stability AI API (CPU Friendly)", "Local SDXL (Requires GPU)"))

topic = st.text_input("Enter a topic for your video:", placeholder="e.g., A brave knight in a magical forest")

if st.button("Generate Video ✨", type="primary"):
    # --- API Key validation ---
    key_needed = {
        "Google Gemini": st.session_state.gemini_key,
        "Hugging Face (Llama 3)": st.session_state.hf_token,
        "Stability AI API (CPU Friendly)": st.session_state.stability_key
    }
    
    if not key_needed.get(text_model_option) or (image_model_option == "Stability AI API (CPU Friendly)" and not key_needed[image_model_option]):
        st.error("Please enter the required API key(s) in the sidebar.", icon="🔑")
    elif not topic:
        st.warning("Please enter a topic to continue.", icon="✍️")
    else:
        with st.status("🎬 Starting video generation...", expanded=True) as status:
            try:
                # 1. Generate Story
                status.update(label="Step 1: Writing the story...")
                story_text = generate_story_with_gemini(topic) if text_model_option == "Google Gemini" else generate_story_with_hf(topic)
                
                # 2. Generate Images
                status.update(label="Step 2: Creating images...")
                sentences = re.split(r'(?<!\w\w.)(?<![A-Z][a-z].)(?<=\.|\?)\s', story_text)
                image_prompts = [sentences[0], sentences[-1]] if len(sentences) >= 2 else [topic, topic]
                image_files = generate_images_api(image_prompts) if image_model_option == "Stability AI API (CPU Friendly)" else generate_images_local(image_prompts)
                
                # 3. Create Video
                status.update(label="Step 3: Assembling the final video...")
                video_file, narration_file = create_video(story_text, image_files)

                status.update(label="✅ Process Complete!", state="complete")
                
                st.subheader("Your AI-Generated Video")
                st.video(open(video_file, 'rb').read())
                st.download_button("Download Video", open(video_file, 'rb').read(), file_name=video_file, mime='video/mp4')
                
                # Clean up temp files
                for f in image_files + [narration_file, video_file]:
                    if os.path.exists(f): os.remove(f)

            except Exception as e:
                st.error(f"An error occurred: {e}", icon="🚨")
