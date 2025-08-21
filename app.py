import streamlit as st
import os
import re
import requests
import base64
import io
import time

# --- Main App Libraries ---
import google.generativeai as genai
from gtts import gTTS
import ffmpeg
from PIL import Image
import numpy as np

# --- Page Configuration ---
st.set_page_config(
    page_title="GEN AI Assessment - II",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 GEN AI Assessment - II")
st.markdown("This application performs the tasks outlined in the Generative AI assessment.")

# --- API Key Management & Model Selection ---
with st.sidebar:
    st.header("⚙️ Model Selection")
    text_model_option = st.selectbox("Choose a Storyteller:", ("Google Gemini", "Hugging Face (Llama 3)"))
    image_model_option = st.selectbox("Choose an Image Artist:", ("Hugging Face (Free Tier)", "Stability AI API"))

    st.header("🔑 API Configuration")
    st.markdown("API keys are required for the selected models.")
    
    # Conditional API Key Inputs
    if text_model_option == "Google Gemini":
        st.session_state.gemini_key = st.text_input(
            label="Google Gemini API Key", type="password", placeholder="Paste Gemini key here...",
            help="[Get your key from Google AI Studio](https://aistudio.google.com/apikey)"
        )
    if "Hugging Face" in text_model_option or "Hugging Face" in image_model_option:
        st.session_state.hf_token = st.text_input(
            label="Hugging Face API Token", type="password", placeholder="Paste Hugging Face token here...",
            help="[Get your token from Hugging Face Settings](https://huggingface.co/settings/tokens)"
        )

    if image_model_option == "Stability AI API":
        st.session_state.stability_key = st.text_input(
            label="Stability AI API Key", type="password", placeholder="Paste Stability AI key here...",
            help="[Get your key from Stability AI](https://platform.stability.ai/)"
        )


# --- Helper Functions ---

# --- CORRECTED: This function now returns the stream iterator ---
def get_gemini_stream(topic):
    genai.configure(api_key=st.session_state.gemini_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
    prompt = f"'{topic}'"
    return model.generate_content(prompt, stream=True)

def generate_story_with_hf(topic):
    API_URL = "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B-Instruct"
    headers = {"Authorization": f"Bearer {st.session_state.hf_token}"}
    prompt = f"'{topic}'"
    response = requests.post(API_URL, headers=headers, json={"inputs": prompt, "parameters": {"max_new_tokens": 512, "return_full_text": False}})
    if response.status_code != 200:
        raise ConnectionError(f"Hugging Face API Error: {response.text}")
    return response.json()[0]['generated_text']

def generate_images_with_hf_api(image_prompts):
    API_URL = "https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5"
    headers = {"Authorization": f"Bearer {st.session_state.hf_token}"}
    image_files = []
    
    for i, prompt in enumerate(image_prompts):
        st.write(f"🖼️ Creating image for: '{prompt[:50]}...'")
        styled_prompt = f"{prompt}, cinematic, masterpiece, high detail"
        response = requests.post(API_URL, headers=headers, json={"inputs": styled_prompt})
        
        if response.status_code == 503: # Model is loading
            estimated_time = response.json().get("estimated_time", 20)
            with st.spinner(f"Model is loading, please wait... (est. {int(estimated_time)}s)"):
                time.sleep(estimated_time)
            response = requests.post(API_URL, headers=headers, json={"inputs": styled_prompt})

        if response.status_code != 200:
            raise ConnectionError(f"Hugging Face Image API Error: {response.text}")

        filename = f"generated_image_{i}.png"
        with open(filename, "wb") as f:
            f.write(response.content)
        image_files.append(filename)
    return image_files

def generate_images_api(image_prompts):
    image_files = []
    for i, prompt in enumerate(image_prompts):
        st.write(f"🖼️ Creating image for: '{prompt[:50]}...'")
        styled_prompt = f"{prompt}, cinematic, masterpiece, high detail"
        response = requests.post(
            "https://api.stability.ai/v1/generation/stable-diffusion-v1-6/text-to-image",
            headers={"Authorization": f"Bearer {st.session_state.stability_key}", "Accept": "application/json"},
            json={"text_prompts": [{"text": styled_prompt}], "samples": 1, "steps": 30, "height": 1024, "width": 1024}
        )
        if response.status_code != 200:
            raise ConnectionError(f"Stability AI API Error: {response.text}")
        data = response.json()
        filename = f"generated_image_{i}.png"
        with open(filename, "wb") as f:
            f.write(base64.b64decode(data["artifacts"][0]["base64"]))
        image_files.append(filename)
    return image_files

def create_video_with_ffmpeg(story_text, image_files):
    narration_file = "narration.mp3"
    tts = gTTS(story_text, lang='en')
    tts.save(narration_file)

    audio_info = ffmpeg.probe(narration_file)
    duration = float(audio_info['format']['duration'])
    duration_per_image = duration / len(image_files)

    image_inputs = [ffmpeg.input(img, t=duration_per_image, loop=1, framerate=24) for img in image_files]
    video_stream = ffmpeg.concat(*image_inputs, v=1, a=0)
    audio_stream = ffmpeg.input(narration_file)

    video_filename = "final_video.mp4"
    (
        ffmpeg
        .output(video_stream, audio_stream, video_filename, vcodec='libx264', acodec='aac', pix_fmt='yuv420p')
        .overwrite_output()
        .run(quiet=True)
    )
    return video_filename, narration_file

# --- Main App Interface ---
## ASSESSMENT STEP 1: Takes a problem statement from the user ##
topic = st.text_input("1. Enter your problem statement / topic:", placeholder="e.g., A robot discovering a garden")

if st.button("Start operation✨", type="primary"):
    # API Key Validation
    key_needed_msg = ""
    if text_model_option == "Google Gemini" and not st.session_state.get('gemini_key'):
        key_needed_msg = "Please enter your Google Gemini API key."
    elif "Hugging Face" in text_model_option and not st.session_state.get('hf_token'):
        key_needed_msg = "Please enter your Hugging Face API token."
    
    if image_model_option == "Stability AI API" and not st.session_state.get('stability_key'):
        key_needed_msg = "Please enter your Stability AI API key."
    elif "Hugging Face" in image_model_option and not st.session_state.get('hf_token'):
        key_needed_msg = "Please enter your Hugging Face API token."

    if key_needed_msg:
        st.error(key_needed_msg, icon="🔑")
    elif not topic:
        st.warning("Please enter a topic to continue.", icon="✍️")
    else:
        with st.status("🎬 Starting generation process...", expanded=True) as status:
            try:
                ## ASSESSMENT STEP 2: Creates a short story / write-up ##
                status.update(label="Step 2: Creating story...")
                story_container = st.empty()
                with story_container.container():
                    st.subheader("📝 2. Generated Story / Write-up")
                    
                    # --- FIX: Handle streaming and non-streaming models correctly ---
                    if text_model_option == "Google Gemini":
                        # Manually display the stream and build the final string
                        story_stream = get_gemini_stream(topic)
                        text_area = st.empty()
                        story_text_parts = [chunk.text for chunk in story_stream]
                        story_text = "".join(story_text_parts)
                        text_area.markdown(story_text)
                    else: # Hugging Face
                        story_text = generate_story_with_hf(topic)
                        st.markdown(story_text)
                
                ## ASSESSMENT STEP 3: Creates a set of images to support step-2 ##
                status.update(label="Step 3: Creating images...")
                sentences = re.split(r'(?<!\w\w.)(?<![A-Z][a-z].)(?<=\.|\?)\s', story_text)
                image_prompts = [sentences[0], sentences[-1]] if len(sentences) >= 2 else [topic, topic]
                
                if image_model_option == "Hugging Face (Free Tier)":
                    image_files = generate_images_with_hf_api(image_prompts)
                else: # Stability AI
                    image_files = generate_images_api(image_prompts)
                
                st.info("🖼️ 3. Supporting Images Generated")
                cols = st.columns(len(image_files))
                for i, img_file in enumerate(image_files):
                    cols[i].image(img_file, caption=f"Image {i+1}")

                ## ASSESSMENT STEP 4: Creates a visual (audio+video) using steps 2 and 3 ##
                status.update(label="Step 4: Creating visual (audio+video)...")
                video_file, narration_file = create_video_with_ffmpeg(story_text, image_files)

                status.update(label="✅ Process Complete!", state="complete")
                
                st.subheader("🎉 4. Final Visual (Audio + Video)")
                st.video(open(video_file, 'rb').read())
                st.download_button("Download Video", open(video_file, 'rb').read(), file_name=video_file, mime='video/mp4')
                
                # Clean up temp files
                for f in image_files + [narration_file, video_file]:
                    if os.path.exists(f): os.remove(f)

            except Exception as e:
                st.error(f"An error occurred: {e}", icon="🚨")

## ASSESSMENT FOOTER ##
st.markdown("---")
st.markdown("<p style='text-align: center;'>Crafted by Shreyas Kasture</p>", unsafe_allow_html=True)
