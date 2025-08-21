# 🎬 VidScribe-AI

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)

An AI-powered Streamlit application that transforms a single text prompt into a short video, complete with AI-generated narration and images.



## ✨ Features

-   **📝 Text-to-Story**: Uses the Google Gemini 1.5 Pro API to generate a creative short story from a simple topic.
-   **🖼️ Story-to-Image**: Creates high-quality images using Stable Diffusion XL (Base + Refiner) based on the generated story.
-   **🗣️ Text-to-Speech**: Converts the story into a natural-sounding audio narration.
-   **🎥 Automatic Video Assembly**: Combines the generated images and narration into a downloadable MP4 video.
-   **⚙️ Memory Optimized**: Loads heavy AI models sequentially to run on hardware with limited VRAM (like a free Google Colab T4 GPU).

---

## 🛠️ Tech Stack

-   **Frontend**: [Streamlit](https://streamlit.io/)
-   **Language Model**: [Google Gemini API](https://ai.google.dev/)
-   **Image Generation**: [Hugging Face Diffusers](https://huggingface.co/docs/diffusers/index) (Stable Diffusion XL)
-   **Audio Generation**: [gTTS](https://pypi.org/project/gTTS/)
-   **Video Processing**: [MoviePy](https://zulko.github.io/moviepy/)

---

## 🚀 Setup and Installation

### Prerequisites

-   Python 3.9+
-   An NVIDIA GPU with CUDA installed (for image generation)
-   A Google Gemini API Key

### Installation Steps

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/VidScribe-AI.git](https://github.com/your-username/VidScribe-AI.git)
    cd VidScribe-AI
    ```

2.  **Install the required Python packages:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up your Gemini API Key:**
    You have two options:
    -   **(Recommended for Deployment)**: Create a file at `.streamlit/secrets.toml` and add your key:
        ```toml
        # .streamlit/secrets.toml
        GEMINI_API_KEY = "YOUR_SECRET_API_KEY_HERE"
        ```
    -   **(For Local Testing)**: Run the app and paste your key into the text input in the sidebar.

4.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```
The application will open in your web browser.

---

## ⚖️ License

This project is licensed under the MIT License. See the `LICENSE` file for details.
