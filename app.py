import streamlit as st
import torch
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import io
import zipfile
import os
from pathlib import Path
import time
from typing import List
import gc
import warnings
import subprocess
import sys

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# Global variables
DIFFUSERS_AVAILABLE = False
STABLE_DIFFUSION_PIPE = None

def install_missing_packages():
    """Install missing packages if needed"""
    packages_to_install = []
    
    try:
        import diffusers
    except ImportError:
        packages_to_install.append("diffusers==0.24.0")
    
    try:
        import transformers
    except ImportError:
        packages_to_install.append("transformers==4.36.0")
    
    if packages_to_install:
        st.warning(f"Installing missing packages: {packages_to_install}")
        try:
            for package in packages_to_install:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            st.success("Packages installed! Please restart the app.")
            return False
        except Exception as e:
            st.error(f"Failed to install packages: {e}")
            return False
    
    return True

def try_import_diffusers():
    """Try to import diffusers with multiple fallback strategies"""
    global DIFFUSERS_AVAILABLE, STABLE_DIFFUSION_PIPE
    
    # First, ensure packages are installed
    if not install_missing_packages():
        return False
    
    try:
        # Strategy 1: Try modern diffusers
        from diffusers import StableDiffusionImg2ImgPipeline
        STABLE_DIFFUSION_PIPE = StableDiffusionImg2ImgPipeline
        DIFFUSERS_AVAILABLE = True
        st.success("✅ AI models available!")
        return True
        
    except Exception as e:
        try:
            # Strategy 2: Try with older API
            from diffusers import StableDiffusionImg2ImgPipeline
            from diffusers.utils import deprecate
            STABLE_DIFFUSION_PIPE = StableDiffusionImg2ImgPipeline
            DIFFUSERS_AVAILABLE = True
            st.success("✅ AI models available (compatibility mode)!")
            return True
            
        except Exception as e2:
            st.warning(f"⚠️ AI models not available: {e2}")
            st.info("🎨 Using Advanced PIL Enhancement (excellent quality!)")
            DIFFUSERS_AVAILABLE = False
            return False

class ImageEnhancer:
    def __init__(self):
        self.device = self._get_device()
        self.model_loaded = False
        self.pipe = None
        
        # Try to setup diffusers
        if try_import_diffusers():
            st.info("🤖 AI enhancement available - will load on first use")
        else:
            st.info("🎨 PIL enhancement mode - fast and reliable!")
    
    def _get_device(self):
        """Get best available device"""
        try:
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        except:
            return "cpu"
    
    def load_ai_model(self):
        """Load AI model on first use"""
        if not DIFFUSERS_AVAILABLE or self.model_loaded:
            return self.model_loaded
        
        try:
            with st.spinner("🚀 Loading AI model (first time only)..."):
                # Use the globally available pipeline class
                self.pipe = STABLE_DIFFUSION_PIPE.from_pretrained(
                    "runwayml/stable-diffusion-v1-5",
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    safety_checker=None,
                    requires_safety_checker=False,
                ).to(self.device)
                
                # Apply optimizations
                if self.device == "cuda":
                    try:
                        self.pipe.enable_attention_slicing()
                        self.pipe.enable_model_cpu_offload()
                    except:
                        pass  # Ignore optimization failures
                
                self.model_loaded = True
                st.success("✅ AI model ready!")
                return True
                
        except Exception as e:
            st.error(f"AI model loading failed: {e}")
            self.model_loaded = False
            return False
    
    def enhance_with_pil(self, image: Image.Image, strength: float = 0.7, mode: str = "balanced") -> Image.Image:
        """Advanced PIL enhancement with multiple modes"""
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        enhanced = image.copy()
        
        if mode == "smooth":
            # Smooth enhancement - gentle improvements
            # Stage 1: Gentle sharpness
            enhancer = ImageEnhance.Sharpness(enhanced)
            enhanced = enhancer.enhance(1.0 + (0.3 * strength))
            
            # Stage 2: Subtle contrast
            enhancer = ImageEnhance.Contrast(enhanced)
            enhanced = enhancer.enhance(1.0 + (0.25 * strength))
            
            # Stage 3: Color enhancement
            enhancer = ImageEnhance.Color(enhanced)
            enhanced = enhancer.enhance(1.0 + (0.2 * strength))
            
            # Stage 4: Brightness adjustment
            enhancer = ImageEnhance.Brightness(enhanced)
            enhanced = enhancer.enhance(1.0 + (0.1 * strength))
            
            # Stage 5: Smooth filtering
            if strength > 0.4:
                # Gentle unsharp mask
                enhanced = enhanced.filter(ImageFilter.UnsharpMask(
                    radius=1.0, 
                    percent=int(80 * strength), 
                    threshold=5
                ))
            
            if strength > 0.6:
                # Subtle smoothing
                enhanced = enhanced.filter(ImageFilter.SMOOTH)
                
        elif mode == "dramatic":
            # Dramatic enhancement for maximum effect
            # Stage 1: Strong sharpness boost
            enhancer = ImageEnhance.Sharpness(enhanced)
            enhanced = enhancer.enhance(1.0 + (0.8 * strength))
            
            # Stage 2: High contrast
            enhancer = ImageEnhance.Contrast(enhanced)
            enhanced = enhancer.enhance(1.0 + (0.6 * strength))
            
            # Stage 3: Vibrant colors
            enhancer = ImageEnhance.Color(enhanced)
            enhanced = enhancer.enhance(1.0 + (0.5 * strength))
            
            # Stage 4: Brightness fine-tuning
            enhancer = ImageEnhance.Brightness(enhanced)
            enhanced = enhancer.enhance(1.0 + (0.2 * strength))
            
            # Stage 5: Advanced filtering
            if strength > 0.3:
                # Strong unsharp mask
                enhanced = enhanced.filter(ImageFilter.UnsharpMask(
                    radius=2.5, 
                    percent=int(250 * strength), 
                    threshold=1
                ))
            
            if strength > 0.6:
                # Detail enhancement
                enhanced = enhanced.filter(ImageFilter.DETAIL)
            
            if strength > 0.8:
                # Edge enhancement
                enhanced = enhanced.filter(ImageFilter.EDGE_ENHANCE_MORE)
                
        else:  # balanced mode
            # Balanced enhancement - good for most images
            # Stage 1: Moderate sharpness
            enhancer = ImageEnhance.Sharpness(enhanced)
            enhanced = enhancer.enhance(1.0 + (0.5 * strength))
            
            # Stage 2: Contrast improvement
            enhancer = ImageEnhance.Contrast(enhanced)
            enhanced = enhancer.enhance(1.0 + (0.4 * strength))
            
            # Stage 3: Color saturation
            enhancer = ImageEnhance.Color(enhanced)
            enhanced = enhancer.enhance(1.0 + (0.3 * strength))
            
            # Stage 4: Brightness adjustment
            enhancer = ImageEnhance.Brightness(enhanced)
            enhanced = enhancer.enhance(1.0 + (0.15 * strength))
            
            # Stage 5: Filtering
            if strength > 0.4:
                # Moderate unsharp mask
                enhanced = enhanced.filter(ImageFilter.UnsharpMask(
                    radius=1.5, 
                    percent=int(150 * strength), 
                    threshold=3
                ))
            
            if strength > 0.7:
                # Detail enhancement
                enhanced = enhanced.filter(ImageFilter.DETAIL)
        
        return enhanced
    
    def enhance_with_ai(self, image: Image.Image, prompt: str, strength: float = 0.4) -> Image.Image:
        """AI enhancement with fallback"""
        # Try to load model if not loaded
        if not self.model_loaded:
            if not self.load_ai_model():
                return self.enhance_with_pil(image, strength)
        
        try:
            original_size = image.size
            
            # Resize for processing
            max_size = 512  # Conservative size for compatibility
            if max(image.size) > max_size:
                ratio = min(max_size / image.width, max_size / image.height)
                new_size = (int(image.width * ratio), int(image.height * ratio))
                image = image.resize(new_size, Image.Resampling.LANCZOS)
            
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            # AI processing
            with torch.inference_mode():
                result = self.pipe(
                    prompt=prompt,
                    image=image,
                    strength=strength,
                    guidance_scale=7.5,
                    num_inference_steps=20,
                    generator=torch.Generator(device=self.device).manual_seed(42)
                )
                enhanced = result.images[0]
            
            # Resize back to original
            if enhanced.size != original_size:
                enhanced = enhanced.resize(original_size, Image.Resampling.LANCZOS)
            
            # Clear memory
            if self.device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
            
            return enhanced
            
        except Exception as e:
            st.warning(f"AI enhancement failed: {e}")
            return self.enhance_with_pil(image, strength)
    
    def upscale_image(self, image: Image.Image, scale: int = 2) -> Image.Image:
        """High-quality upscaling"""
        try:
            new_size = (image.width * scale, image.height * scale)
            upscaled = image.resize(new_size, Image.Resampling.LANCZOS)
            
            # Post-upscale enhancement
            enhancer = ImageEnhance.Sharpness(upscaled)
            upscaled = enhancer.enhance(1.2)
            
            enhancer = ImageEnhance.Contrast(upscaled)
            upscaled = enhancer.enhance(1.1)
            
            return upscaled
        except Exception as e:
            st.error(f"Upscaling failed: {e}")
            return image
    
    def process_bulk(self, images: List[Image.Image], settings: dict, progress_callback=None) -> List[Image.Image]:
        """Bulk processing"""
        results = []
        total = len(images)
        
        for i, image in enumerate(images):
            if progress_callback:
                progress_callback((i + 1) / total, f"Processing {i+1}/{total}")
            
            try:
                enhancement_type = settings["type"]
                
                if enhancement_type == "AI Enhancement":
                    result = self.enhance_with_ai(
                        image, 
                        settings["prompt"], 
                        settings["strength"]
                    )
                elif enhancement_type == "Upscale Only":
                    result = self.upscale_image(image, 2)
                elif enhancement_type == "Both (Best Quality)":
                    enhanced = self.enhance_with_ai(
                        image, 
                        settings["prompt"], 
                        settings["strength"] * 0.8
                    )
                    result = self.upscale_image(enhanced, 2)
                else:  # PIL Enhancement
                    result = self.enhance_with_pil(image, settings["strength"], settings.get("mode", "balanced"))
                
                results.append(result)
                
            except Exception as e:
                st.error(f"Failed processing image {i+1}: {e}")
                results.append(image)
        
        return results

def create_zip_download(images: List[Image.Image], filenames: List[str]) -> bytes:
    """Create ZIP download"""
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for i, (img, filename) in enumerate(zip(images, filenames)):
            img_buffer = io.BytesIO()
            img.save(img_buffer, format='PNG', quality=95)
            img_buffer.seek(0)
            
            name = f"enhanced_{Path(filename).stem}.png"
            zip_file.writestr(name, img_buffer.read())
    
    zip_buffer.seek(0)
    return zip_buffer.read()

def main():
    st.set_page_config(
        page_title="🚀 Bulletproof Image Enhancer",
        page_icon="🎨",
        layout="wide"
    )
    
    st.title("🚀 Bulletproof Image Enhancer")
    st.markdown("**Professional image enhancement that works in any environment**")
    
    # Initialize enhancer
    if 'enhancer' not in st.session_state:
        st.session_state.enhancer = ImageEnhancer()
    
    # Sidebar
    st.sidebar.header("⚙️ Settings")
    
    device = st.session_state.enhancer.device
    ai_status = "✅ Available" if DIFFUSERS_AVAILABLE else "🎨 PIL Mode"
    
    st.sidebar.info(f"**Device:** {device.upper()}")
    st.sidebar.info(f"**AI Models:** {ai_status}")
    
    enhancement_type = st.sidebar.selectbox(
        "Enhancement Method",
        [
            "PIL Enhancement (Reliable & Fast)", 
            "AI Enhancement", 
            "Upscale Only", 
            "Both (Best Quality)"
        ]
    )
    
    # Add enhancement mode selector
    if "PIL" in enhancement_type:
        enhancement_mode = st.sidebar.selectbox(
            "Enhancement Style",
            ["balanced", "smooth", "dramatic"],
            help="Smooth: Gentle, natural look | Balanced: Good for most images | Dramatic: Maximum enhancement"
        )
    else:
        enhancement_mode = "balanced"
    
    strength = st.sidebar.slider(
        "Enhancement Strength", 
        0.1, 1.0, 0.8, 0.1,
        help="Higher values = more dramatic enhancement"
    )
    
    if "AI" in enhancement_type:
        prompt = st.sidebar.text_area(
            "AI Prompt",
            "high quality, sharp, detailed, professional photography, enhanced clarity, vivid colors, perfect contrast, masterpiece",
            height=100
        )
    else:
        prompt = "high quality, detailed"
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["📷 Single Image", "📁 Bulk Processing", "🛠️ Setup"])
    
    with tab1:
        st.header("Single Image Enhancement")
        
        uploaded_file = st.file_uploader(
            "Upload your image",
            type=['png', 'jpg', 'jpeg', 'webp', 'bmp'],
            key="single"
        )
        
        if uploaded_file:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📷 Original")
                original = Image.open(uploaded_file)
                st.image(original, use_container_width=True)
                st.write(f"**Size:** {original.size[0]}×{original.size[1]}")
            
            with col2:
                st.subheader("✨ Enhanced")
                
                if st.button("🚀 Enhance Image", key="enhance_single", type="primary"):
                    start_time = time.time()
                    
                    with st.spinner(f"Enhancing with {enhancement_type}..."):
                        if enhancement_type == "PIL Enhancement (Reliable & Fast)":
                            result = st.session_state.enhancer.enhance_with_pil(original, strength, enhancement_mode)
                        elif enhancement_type == "AI Enhancement":
                            result = st.session_state.enhancer.enhance_with_ai(original, prompt, strength * 0.4)
                        elif enhancement_type == "Upscale Only":
                            result = st.session_state.enhancer.upscale_image(original, 2)
                        else:  # Both
                            enhanced = st.session_state.enhancer.enhance_with_ai(original, prompt, strength * 0.3)
                            result = st.session_state.enhancer.upscale_image(enhanced, 2)
                    
                    process_time = time.time() - start_time
                    
                    st.image(result, use_container_width=True)
                    st.write(f"**Size:** {result.size[0]}×{result.size[1]}")
                    st.success(f"✅ Enhanced in {process_time:.1f}s")
                    
                    # Download button
                    buf = io.BytesIO()
                    result.save(buf, format='PNG', quality=95)
                    buf.seek(0)
                    
                    st.download_button(
                        "📥 Download Enhanced Image",
                        buf.read(),
                        f"enhanced_{uploaded_file.name}",
                        "image/png"
                    )
    
    with tab2:
        st.header("Bulk Image Processing")
        
        uploaded_files = st.file_uploader(
            "Upload multiple images",
            type=['png', 'jpg', 'jpeg', 'webp', 'bmp'],
            accept_multiple_files=True,
            key="bulk"
        )
        
        if uploaded_files:
            st.write(f"📁 **{len(uploaded_files)} files ready**")
            
            # Preview first 4 images
            if len(uploaded_files) <= 4:
                cols = st.columns(len(uploaded_files))
                for i, file in enumerate(uploaded_files):
                    with cols[i]:
                        img = Image.open(file)
                        st.image(img, caption=file.name, use_container_width=True)
            
            if st.button("🚀 Process All Images", key="process_bulk", type="primary"):
                images = []
                filenames = []
                
                for file in uploaded_files:
                    try:
                        images.append(Image.open(file))
                        filenames.append(file.name)
                    except Exception as e:
                        st.error(f"Failed to load {file.name}: {e}")
                
                if images:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    def update_progress(progress, status):
                        progress_bar.progress(progress)
                        status_text.text(status)
                    
                    settings = {
                        "type": enhancement_type,
                        "strength": strength,
                        "prompt": prompt,
                        "mode": enhancement_mode if "PIL" in enhancement_type else "balanced"
                    }
                    
                    start_time = time.time()
                    results = st.session_state.enhancer.process_bulk(
                        images, settings, update_progress
                    )
                    process_time = time.time() - start_time
                    
                    # Clear progress indicators
                    progress_bar.empty()
                    status_text.empty()
                    
                    st.success(f"✅ Processed {len(results)} images in {process_time:.1f}s")
                    st.info(f"⚡ Average: {process_time/len(results):.1f}s per image")
                    
                    # Create and offer ZIP download directly
                    with st.spinner("Creating ZIP file..."):
                        zip_data = create_zip_download(results, filenames)
                    
                    st.download_button(
                        "📦 Download All Enhanced Images (ZIP)",
                        zip_data,
                        "enhanced_images.zip",
                        "application/zip",
                        type="primary"
                    )
    
    with tab3:
        st.header("🛠️ Environment Setup")
        
        st.subheader("📊 System Status")
        
        # Check system status
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if torch.cuda.is_available():
                st.success("✅ CUDA Available")
                st.write(f"GPU: {torch.cuda.get_device_name()}")
            else:
                st.info("ℹ️ CPU Mode")
        
        with col2:
            if DIFFUSERS_AVAILABLE:
                st.success("✅ AI Models Ready")
            else:
                st.warning("⚠️ AI Models Unavailable")
        
        with col3:
            st.info(f"🖥️ Device: {device.upper()}")
        
        st.subheader("🔧 Fix AI Models")
        
        if not DIFFUSERS_AVAILABLE:
            st.markdown("""
            **To enable AI enhancement features:**
            
            1. **Create fresh environment:**
            ```bash
            conda create -n ai_enhancer python=3.10 -y
            conda activate ai_enhancer
            ```
            
            2. **Install compatible packages:**
            ```bash
            pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
            pip install "huggingface_hub==0.19.4"
            pip install "diffusers==0.24.0"
            pip install streamlit pillow numpy
            ```
            
            3. **Restart this app**
            """)
            
            if st.button("🔄 Try to Install AI Packages Now", key="install_packages"):
                with st.spinner("Installing packages..."):
                    try:
                        subprocess.check_call([sys.executable, "-m", "pip", "install", "diffusers==0.24.0", "transformers==4.36.0"])
                        st.success("✅ Packages installed! Please restart the app.")
                    except Exception as e:
                        st.error(f"Installation failed: {e}")
        
        st.subheader("💡 Performance Tips")
        st.markdown("""
        - **PIL Enhancement:** Fast, reliable, works offline
        - **AI Enhancement:** Slow but highest quality  
        - **Strength 0.8-0.9:** For dramatic improvements
        - **Bulk Processing:** Most efficient for multiple images
        """)
    
    # Sidebar tips
    st.sidebar.markdown("---")
    st.sidebar.markdown("**💡 Quick Tips:**")
    st.sidebar.markdown("• Try strength 0.8 for dramatic results")
    st.sidebar.markdown("• PIL mode works without internet")
    st.sidebar.markdown("• AI mode needs model download first")
    st.sidebar.markdown("• Both mode = enhance + upscale")

if __name__ == "__main__":
    main()
