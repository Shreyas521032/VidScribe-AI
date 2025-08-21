import streamlit as st
from gtts import gTTS
from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips
import google.generativeai as genai
import os
import requests
from PIL import Image
import numpy as np
import io

# --- Page Configuration ---
st.set_page_config(
    page_title="VidScribe AI (Optimized)",
    page_icon="⚡",
    layout="centered"
)

st.title("⚡ VidScribe AI (Optimized)")
st.markdown("An optimized version that streams text and processes media faster.")

# --- API Key Management ---
st.sidebar.header("🔑 API Configuration")
if 'api_key' not in st.session_state:
    st.session_state.api_key = ''

st.session_state.api_key = st.sidebar.text_input(
    label="Enter your Google Gemini API Key", type="password", 
    placeholder="Paste your key here...", value=st.session_state.api_key
)

# --- Helper Functions (Optimized) ---

def generate_story_stream(topic):
    """OPTIMIZATION: Generates a story using the Gemini API's streaming feature."""
    if not st.session_state.api_key:
        st.error("Please enter your Gemini API Key in the sidebar.")
        return
        
    genai.configure(api_key=st.session_state.api_key)
    model = genai.GenerativeModel('gemini-1.5-flash') # Using Flash for speed
    prompt = f"Write a captivating short story about '{topic}'. The story should be about 3-4 paragraphs long."
    
    # Stream the response
    response_stream = model.generate_content(prompt, stream=True)
    
    # Use st.write_stream to display the content as it arrives
    return st.write_stream(response_stream)

def generate_images_in_memory(image_prompts):
    """
    OPTIMIZATION: Generates images using a generic API and processes them in memory.
    This function is a placeholder for any image generation API call.
    """
    image_arrays = []
    
    # Placeholder: Replace with your actual image generation API call
    # For this example, we'll download placeholder images.
    for i, prompt in enumerate(image_prompts):
        st.write(f"🖼️ Creating image for: '{prompt[:50]}...'")
        # Replace this URL with your API endpoint
        response = requests.get(f"https://picsum.photos/1024/1024?random={i}")
        
        if response.status_code == 200:
            # OPTIMIZATION: Convert image bytes to a numpy array directly in memory
            image_bytes = io.BytesIO(response.content)
            pil_image = Image.open(image_bytes)
            image_arrays.append(np.array(pil_image))
        else:
            st.error(f"Failed to fetch placeholder image. Status: {response.status_code}")

    return image_arrays

def create_video_fast(story_text, image_arrays):
    """OPTIMIZATION: Creates video with faster encoding settings."""
    # 1. Create Audio (gTTS requires file I/O)
    narration_file = "narration.mp3"
    try:
        tts = gTTS(story_text, lang='en')
        tts.save(narration_file)
        audioclip = AudioFileClip(narration_file)
    except Exception as e:
        st.error(f"Failed to create audio: {e}")
        return None, None

    # 2. Create Video Clips from in-memory image arrays
    duration_per_image = audioclip.duration / len(image_arrays)
    clips = [ImageClip(img_array).set_duration(duration_per_image) for img_array in image_arrays]
    
    # 3. Assemble and Write Video File
    slideshow = concatenate_videoclips(clips, method="compose")
    final_video = slideshow.set_audio(audioclip)
    
    video_filename = "final_video.mp4"
    # OPTIMIZATION: Use a faster preset and multiple threads for encoding
    final_video.write_videofile(
        video_filename, 
        fps=24, 
        codec='libx264',
        preset='ultrafast', # Sacrifices file size for speed
        threads=4          # Use 4 CPU cores
    )

    return video_filename, narration_file

# --- Main App Interface ---
topic = st.text_input("Enter a topic for your video:", placeholder="e.g., A robot discovering a garden")

if st.button("Generate Video ✨", type="primary"):
    if not st.session_state.api_key:
        st.error("Please enter your Gemini API key in the sidebar.", icon="🔑")
    elif not topic:
        st.warning("Please enter a topic to continue.", icon="✍️")
    else:
        # Use a single status context manager for the whole process
        with st.status("🎬 Starting video generation...", expanded=True) as status:
            try:
                # Step 1: Generate and stream the story
                status.update(label="Step 1: Writing the story with Gemini...")
                story_container = st.empty()
                with story_container.container():
                    st.subheader("📝 Your AI-Generated Story")
                    story_text = generate_story_stream(topic)
                
                # Step 2: Generate images
                status.update(label="Step 2: Creating images...")
                # Using placeholder prompts for this example
                image_prompts = [f"A cinematic photo of {topic}", f"An artistic painting of {topic}"]
                image_arrays = generate_images_in_memory(image_prompts)
                
                # Step 3: Create the video
                status.update(label="Step 3: Assembling the final video...")
                video_file, narration_file = create_video_fast(story_text, image_arrays)

                status.update(label="✅ Process Complete!", state="complete")
                
                # Display final video
                st.subheader("🎉 Your AI-Generated Video")
                st.video(open(video_file, 'rb').read())
                st.download_button("Download Video", open(video_file, 'rb').read(), file_name=video_file, mime='video/mp4')
                
                # Clean up temp files
                for f in [narration_file, video_file]:
                    if f and os.path.exists(f):
                        os.remove(f)

            except Exception as e:
                st.error(f"An error occurred: {e}", icon="🚨")
