import streamlit as st
import torch
from diffusers import AutoPipelineForText2Image
from gtts import gTTS
from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips
import re
import gc
import google.generativeai as genai
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="VidScribe AI",
    page_icon="🎬",
    layout="centered"
)

st.title("🎬 VidScribe AI")
st.markdown("Turn any topic into a short video with AI-generated narration and images.")

# --- API Key Management (UPDATED) ---
st.sidebar.header("API Configuration")

# Try to get the key from Streamlit's secrets
try:
    api_key_secret = st.secrets["GEMINI_API_KEY"]
except (FileNotFoundError, KeyError):
    api_key_secret = None

# If the key is not in secrets, ask the user for it
user_api_key = st.sidebar.text_input(
    label="Enter your Gemini API Key",
    type="password",
    placeholder="Paste your key here...",
    help="You can get your key from Google AI Studio.",
    value=api_key_secret or st.session_state.get("api_key", "")
)

# Store the key in session state
st.session_state.api_key = user_api_key

# Configure the Gemini client if the key is available
if st.session_state.api_key:
    try:
        genai.configure(api_key=st.session_state.api_key)
        st.sidebar.success("✅ Gemini API key configured.")
    except Exception as e:
        st.sidebar.error("🚨 Invalid API Key. Please check your key.")
        st.stop()
else:
    st.sidebar.warning("Please enter your Gemini API key to begin.")
    st.stop()


# --- Helper Functions (Our Script Logic) ---
@st.cache_resource
def get_gemini_model():
    """Loads the Gemini model, cached for performance."""
    return genai.GenerativeModel('gemini-1.5-pro-latest')

@st.cache_data
def generate_story(topic):
    """Generates a story using the Gemini API."""
    model = get_gemini_model()
    prompt = f"You are a master storyteller. Write a captivating and imaginative short story about '{topic}'. The story should have rich descriptions, a clear narrative arc, and be approximately 3-4 paragraphs long."
    response = model.generate_content(prompt)
    return response.text

@st.cache_resource(show_spinner="Loading image generation models...")
def load_diffusion_pipelines():
    """Loads and caches the SDXL models."""
    base = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16, variant="fp16", use_safensors=True
    ).to("cuda")
    refiner = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        torch_dtype=torch.float16, use_safensors=True, variant="fp16",
    ).to("cuda")
    return base, refiner

def generate_images(_image_prompts):
    """Generates images sequentially to save memory."""
    image_files = []
    base, refiner = load_diffusion_pipelines() # Load cached models
    
    latent_images = []
    for prompt in _image_prompts:
        styled_prompt = f"{prompt}, masterpiece, 8k, high detail, cinematic lighting"
        negative_prompt = "ugly, blurry, deformed, disfigured, poor details, bad anatomy"
        latent = base(prompt=styled_prompt, negative_prompt=negative_prompt, output_type="latent").images
        latent_images.append(latent)

    for i, latent in enumerate(latent_images):
        prompt = _image_prompts[i]
        styled_prompt = f"{prompt}, masterpiece, 8k, high detail, cinematic lighting"
        negative_prompt = "ugly, blurry, deformed, disfigured, poor details, bad anatomy"
        
        img = refiner(prompt=styled_prompt, negative_prompt=negative_prompt, image=latent).images[0]
        filename = f"generated_image_{i}.png"
        img.save(filename)
        image_files.append(filename)
        
    return image_files

def create_video(story_text, image_files):
    """Creates audio narration and assembles the final video."""
    # Create audio
    tts = gTTS(story_text, lang='en')
    narration_file = "narration.mp3"
    tts.save(narration_file)

    # Create video
    audioclip = AudioFileClip(narration_file)
    duration_per_image = audioclip.duration / len(image_files)
    
    clips = [ImageClip(img_file).set_duration(duration_per_image) for img_file in image_files]
    
    slideshow = concatenate_videoclips(clips, method="compose")
    final_video = slideshow.set_audio(audioclip)
    
    video_filename = "final_video.mp4"
    final_video.write_videofile(video_filename, fps=24, codec='libx264')

    return video_filename, narration_file

# --- Main App Interface ---
topic = st.text_input("Enter a topic for your video:", placeholder="e.g., A brave knight in a magical forest")

if st.button("Generate Video ✨", type="primary"):
    if not topic:
        st.warning("Please enter a topic to continue.")
    else:
        with st.status("🎬 Starting video generation process...", expanded=True) as status:
            try:
                # 1. Generate Story
                status.update(label="Step 1: Writing the story with Gemini...")
                story_text = generate_story(topic)
                st.info("📝 Story Generated")
                st.markdown(f"> {story_text[:200]}...")

                # 2. Generate Images
                status.update(label="Step 2: Creating images with Stable Diffusion XL...")
                sentences = re.split(r'(?<!\w\w.)(?<![A-Z][a-z].)(?<=\.|\?)\s', story_text)
                sentences = [s.strip() for s in sentences if s.strip()]
                image_prompts = [sentences[0], sentences[-1]] if len(sentences) >= 2 else [topic, topic]
                
                image_files = generate_images(tuple(image_prompts)) # Use tuple for caching
                st.info("🖼️ Images Generated")
                
                cols = st.columns(len(image_files))
                for i, img_file in enumerate(image_files):
                    cols[i].image(img_file, caption=f"Image {i+1}")

                # 3. Create Video
                status.update(label="Step 3: Assembling the final video...")
                video_file, narration_file = create_video(story_text, image_files)
                st.info("🎥 Video Assembled")

                # --- Display Final Video ---
                status.update(label="✅ Process Complete!", state="complete")
                
                st.subheader("Your AI-Generated Video")
                video_bytes = open(video_file, 'rb').read()
                st.video(video_bytes)
                
                st.download_button(
                    label="Download Video",
                    data=video_bytes,
                    file_name=video_file,
                    mime='video/mp4'
                )
                
                # Clean up files
                for f in image_files + [narration_file, video_file]:
                    if os.path.exists(f):
                        os.remove(f)

            except Exception as e:
                st.error(f"An error occurred: {e}", icon="🚨")
