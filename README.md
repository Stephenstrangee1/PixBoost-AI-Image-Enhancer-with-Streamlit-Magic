# Streamlit Image Enhancer

AI Chatbot for UGC NET/CSIR Prep: Instant exam tips, syllabus breakdowns, and enrollment guidance from Professor Academy. 

**Bulletproof Image Enhancer** - Professional tool with AI (Stable Diffusion) fallback to PIL for reliable enhancement. Handles single/bulk images, upscaling, and dramatic effects.

## Features
- PIL Mode: Fast, offline enhancement (sharpness, contrast, color).
- AI Mode: Stable Diffusion for pro-level upscaling/detailing.
- Bulk Processing: ZIP export for multiple files.
- Auto-Setup: Installs missing packages on run.
- Responsive: Works on CPU/GPU/MPS.

## Quick Start
1. Clone: `git clone https://github.com/YOURUSERNAME/streamlit-image-enhancer.git`
2. Install: `pip install -r requirements.txt`
3. Run: `streamlit run app.py`
4. Open: http://localhost:8501

**Pro Tip:** For AI features, ensure PyTorch + CUDA if on GPU. App auto-falls back to PIL.

## Screenshots
<img width="1919" height="1079" alt="2" src="https://github.com/user-attachments/assets/5c7be755-1d5f-43c9-a4c7-85a1438f852d" />


## Tech Stack
- Streamlit for UI
- Diffusers/Transformers for AI
- PIL for fallback enhancement
- Torch for GPU accel

## License
MIT
