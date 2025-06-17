import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import timm
from PIL import Image
import numpy as np
import cv2
import dlib
from typing import Tuple, List, Union, Optional
import tempfile
import os
import time

# Set page config
st.set_page_config(
    page_title="DeepFake Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

class_names = ['Real', 'DeepFake']

class DeepFakeDetector:
    def __init__(self, model_path: str, landmark_predictor_path: str = 'shape_predictor_81_face_landmarks.dat'):
        """
        Initialize the DeepFake detector with a Swin-S model.
        
        Args:
            model_path (str): Path to the model checkpoint
            landmark_predictor_path (str): Path to the dlib facial landmark predictor
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize face detector and landmark predictor
        self.face_detector = dlib.get_frontal_face_detector()
        self.landmark_predictor = dlib.shape_predictor(landmark_predictor_path)
        
        self.model = self._load_model(model_path)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _load_model(self, checkpoint_path: str) -> nn.Module:
        """
        Load the Swin-S model from checkpoint.
        """
        model = timm.create_model('swin_small_patch4_window7_224', pretrained=False, num_classes=2)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(self.device)
        model.eval()
        return model

    def get_face_landmarks(self, image: np.ndarray) -> Optional[dlib.rectangle]:
        """
        Get facial landmarks using dlib's predictor.
        
        Args:
            image: numpy array of the image
            
        Returns:
            dlib.rectangle of the detected face or None if no face is found
        """
        # Convert to grayscale for face detection
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
            
        # Detect faces
        faces = self.face_detector(gray)
        if len(faces) == 0:
            return None
            
        # Get the largest face
        face = max(faces, key=lambda rect: rect.width() * rect.height())
        return face

    def extract_face(self, image: Union[Image.Image, np.ndarray], padding: int = 50) -> Optional[Image.Image]:
        """
        Extract face from an image using dlib face detector and landmark predictor.
        
        Args:
            image: PIL Image or numpy array (in RGB format)
            padding: Padding around the detected face in pixels
            
        Returns:
            PIL Image of the extracted face or None if no face is detected
        """
        # Convert PIL Image to numpy array if necessary
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            # If it's a numpy array, ensure it's RGB
            image_np = image.copy()
            if len(image_np.shape) == 3 and image_np.shape[2] == 3:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            
        # Get face rectangle
        face = self.get_face_landmarks(image_np)
        if face is None:
            return None
            
        # Get facial landmarks (used for face extraction but not shown)
        landmarks = self.landmark_predictor(image_np, face)
        
        # Get face boundaries with padding
        x1 = max(0, face.left() - padding)
        y1 = max(0, face.top() - padding)
        x2 = min(image_np.shape[1], face.right() + padding)
        y2 = min(image_np.shape[0], face.bottom() + padding)
        
        # Extract face region
        face_image = image_np[y1:y2, x1:x2]
        
        # Return as PIL Image in RGB format
        return Image.fromarray(face_image)

    def predict_image(self, image: Union[Image.Image, np.ndarray]) -> Tuple[float, int]:
        """
        Process a single image and return the prediction.
        
        Args:
            image: PIL Image or numpy array
        
        Returns:
            tuple: (prediction probability, binary prediction)
        """
        # Extract face
        face_image = self.extract_face(image)
        if face_image is None:
            return None, None
        
        # Transform image
        image_tensor = self.transform(face_image).unsqueeze(0).to(self.device)
        
        # Get prediction
        with torch.no_grad():
            output = self.model(image_tensor)
            prob = torch.softmax(output, dim=1)[0, 1].cpu().item()
            prediction = 1 if prob >= 0.5 else 0
            
        return prob, prediction

    def extract_frames(self, video: Union[str, cv2.VideoCapture], num_frames: int = 30) -> List[Image.Image]:
        """
        Extract frames from a video and detect faces in each frame.
        
        Args:
            video: OpenCV VideoCapture object or path to video file
            num_frames: Number of frames to extract
        
        Returns:
            List[Image.Image]: List of extracted face images as PIL Images
        """
        # If string path is provided, create VideoCapture object
        if isinstance(video, str):
            cap = cv2.VideoCapture(video)
        else:
            cap = video
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB and create PIL Image directly
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)
                face_image = self.extract_face(frame_pil)
                if face_image is not None:
                    frames.append(face_image)
        
        if isinstance(video, str):
            cap.release()
            
        if not frames:
            raise ValueError("No faces detected in any of the video frames")
            
        return frames

    def predict_video(self, video: Union[str, cv2.VideoCapture, List[Image.Image]]) -> Tuple[float, int, List[Image.Image]]:
        """
        Process a video and return the prediction.
        
        Args:
            video: Video file path, OpenCV VideoCapture object, or list of frames
        
        Returns:
            tuple: (prediction probability, binary prediction, extracted faces)
        """
        # Get frames if not already provided
        if isinstance(video, (str, cv2.VideoCapture)):
            frames = self.extract_frames(video)
        else:
            frames = video

        if not frames:
            raise ValueError("No frames could be extracted from the video")
        
        # Process each frame
        frame_predictions = []
        for frame in frames:
            prob, _ = self.predict_image(frame)
            if prob is not None:
                frame_predictions.append(prob)
        
        # Average the predictions across frames
        avg_prob = np.mean(frame_predictions)
        final_prediction = 1 if avg_prob >= 0.5 else 0
        
        return avg_prob, final_prediction, frames


@st.cache_resource
def load_detector(model_path=None, landmark_path=None):
    """Load the DeepFake detector model (cached for performance)"""
    try:
        # Use provided paths or default paths
        model_file = model_path if model_path else 'models/swin_model.pth'
        landmark_file = landmark_path if landmark_path else 'models/face_extraction_model.dat'
        
        detector = DeepFakeDetector(
            model_path=model_file,
            landmark_predictor_path=landmark_file
        )
        return detector
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None


def main():
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-bottom: 2rem;
        border-radius: 10px;
    }
    .result-container {
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin: 2rem 0;
    }
    .real-result {
        background-color: #d4edda;
        border: 2px solid #28a745;
        color: #155724;
    }
    .fake-result {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
        color: #721c24;
    }
    .upload-section {
        border: 2px dashed #ccc;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    .face-gallery {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        justify-content: center;
        margin: 1rem 0;
    }
    .face-item {
        text-align: center;
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 5px;
    }
    .probability-section {
        display: flex;
        justify-content: space-around;
        margin: 1rem 0;
    }
    .prob-card {
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        min-width: 150px;
    }
    .real-prob {
        background-color: #d4edda;
        border: 2px solid #28a745;
        color: #155724;
    }
    .fake-prob {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
        color: #721c24;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🔍 DeepFake Detection System</h1>
        <p>Upload a video to detect if it contains deepfake content</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("📋 Instructions")
    st.sidebar.markdown("""
    1. **Upload Video**: Choose a video file (MP4, AVI, MOV)
    2. **Processing**: The system will analyze frames and detect faces
    3. **Results**: Get prediction with confidence score
    
    **Supported formats**: MP4, AVI, MOV, MKV
    """)
    
    st.sidebar.header("ℹ️ About")
    st.sidebar.markdown("""
    This system uses a Swin Transformer model to detect deepfake content in videos.
    It analyzes multiple frames and provides an average confidence score.
    """)
    
    # Manual Upload Section
    st.sidebar.header("📤 Manual Upload")
    
    # Model file upload
    st.sidebar.markdown("**Upload Swin model (.pth)**")
    model_file = st.sidebar.file_uploader(
        "Drag and drop file here",
        type=['pth'],
        help="Limit 200MB per file • PTH",
        key="model_upload"
    )
    
    # Landmark predictor upload
    st.sidebar.markdown("**Upload landmark predictor (.dat)**")
    landmark_file = st.sidebar.file_uploader(
        "Drag and drop file here",
        type=['dat'],
        help="Limit 200MB per file • DAT",
        key="landmark_upload"
    )
    
    # Handle manual uploads
    model_path = None
    landmark_path = None
    
    if model_file is not None:
        # Save model file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as tmp_file:
            tmp_file.write(model_file.read())
            model_path = tmp_file.name
        st.sidebar.success("✅ Model uploaded successfully!")
    
    if landmark_file is not None:
        # Save landmark file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.dat') as tmp_file:
            tmp_file.write(landmark_file.read())
            landmark_path = tmp_file.name
        st.sidebar.success("✅ Landmark predictor uploaded successfully!")
    
    # Load detector
    detector = load_detector(model_path, landmark_path)
    
    if detector is None:
        st.error("❌ Failed to load the detection model. Please check model files or upload them manually.")
        return
    
    # File upload section (directly after header, no extra panel)
    st.subheader("📁 Upload Video File")
    
    uploaded_file = st.file_uploader(
        "Choose a video file...",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Upload a video file to analyze for deepfake content"
    )
    
    if uploaded_file is not None:
        # Save uploaded file temporarily for video display
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_video_path = tmp_file.name
        
        # Reset file pointer for later use
        uploaded_file.seek(0)
        
        # Display video info
        st.subheader("📹 Uploaded Video")
        col1, col2 = st.columns([1, 1])  # Made video preview smaller
        
        with col1:
            try:
                # Try to display the video using the temporary file
                with open(temp_video_path, 'rb') as video_file:
                    video_bytes = video_file.read()
                st.video(video_bytes, format="video/mp4", start_time=0)
            except Exception as e:
                st.warning("⚠️ Cannot preview video in browser. Video will still be processed for analysis.")
                st.info("💡 This is normal for some video formats. The detection will work regardless.")
        
        with col2:
            st.markdown(f"""
            **File Details:**
            - Name: {uploaded_file.name}
            - Size: {uploaded_file.size / (1024*1024):.2f} MB
            - Type: {uploaded_file.type}
            """)
        
        # Process video button
        if st.button("🔍 Analyze Video", type="primary", use_container_width=True):
            try:
                # Record start time for analysis timing
                start_time = time.time()
                
                # Show progress
                st.subheader("🔄 Processing Video...")
                
                # Create progress components
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Step 1: Loading video
                status_text.text("📽️ Loading video...")
                progress_bar.progress(20)
                time.sleep(0.5)
                
                # Step 2: Extracting frames
                status_text.text("🎬 Extracting frames...")
                progress_bar.progress(40)
                
                # Create video capture
                cap = cv2.VideoCapture(temp_video_path)
                
                # Step 3: Detecting faces
                status_text.text("👤 Detecting faces...")
                progress_bar.progress(60)
                time.sleep(0.5)
                
                # Step 4: Analyzing frames
                status_text.text("🤖 Analyzing with AI model...")
                progress_bar.progress(80)
                
                # Get prediction with extracted faces
                prob, prediction, extracted_faces = detector.predict_video(cap)
                cap.release()
                
                # Step 5: Complete
                status_text.text("✅ Analysis complete!")
                progress_bar.progress(100)
                time.sleep(0.5)
                
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
                
                # Show results
                st.subheader("📊 Detection Results")
                
                # Show Real and Fake probabilities
                real_prob = 1 - prob
                fake_prob = prob
                
                
                
                
                # Determine result styling
                if prediction == 0:  # Real
                    result_class = "real-result"
                    result_icon = "✅"
                    result_text = "REAL"
                    # Display main result
                    st.markdown(f"""
                    <div class="result-container {result_class}">
                        <h2>{result_icon} {result_text}</h2>
                        <h3>Overall Confidence: {real_prob:.2%}</h3>
                        <p>Final Prediction: {class_names[prediction]}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:  # Fake
                    result_class = "fake-result"
                    result_icon = "⚠️"
                    result_text = "DEEPFAKE"
                     # Display main result
                    st.markdown(f"""
                    <div class="result-container {result_class}">
                        <h2>{result_icon} {result_text}</h2>
                        <h3>Overall Confidence: {fake_prob:.2%}</h3>
                        <p>Final Prediction: {class_names[prediction]}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Show sample extracted faces
                st.subheader("👥 Sample Extracted Faces")
                if extracted_faces:
                    # Show up to 5 faces
                    faces_to_show = extracted_faces[:5]
                    cols = st.columns(5)
                    
                    for i, face in enumerate(faces_to_show):
                        with cols[i]:
                            st.image(face, caption=f"Face {i+1}", use_container_width=True)
                else:
                    st.warning("No faces were extracted from the video.")
                
                # Additional details
                st.subheader("📈 Detailed Analysis")
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    st.metric("Frames Analyzed", len(extracted_faces))
                
                with col2:
                    st.metric("Faces Detected", len(extracted_faces))
                
                with col3:
                    st.metric("Device Used", detector.device.type.upper())
                
                with col4:
                    st.metric("Real Probability", f"{real_prob:.1%}")
                
                with col5:
                    st.metric("Fake Probability", f"{fake_prob:.1%}")
                
                # Confidence interpretation
                st.subheader("🎯 Confidence Interpretation")
                if prob < 0.3:
                    st.success("🟢 **High confidence**: Very likely to be real content")
                elif prob < 0.7:
                    st.warning("🟡 **Medium confidence**: Uncertain, requires human review")
                else:
                    st.error("🔴 **High confidence**: Very likely to be deepfake content")
                
            except ValueError as e:
                st.error(f"❌ **Processing Error**: {str(e)}")
                st.info("💡 **Tip**: Make sure the video contains clear faces that can be detected.")
                
            except Exception as e:
                st.error(f"❌ **Unexpected Error**: {str(e)}")
                
            finally:
                # Clean up temporary files
                if 'temp_video_path' in locals() and os.path.exists(temp_video_path):
                    os.unlink(temp_video_path)
                if model_path and os.path.exists(model_path):
                    os.unlink(model_path)
                if landmark_path and os.path.exists(landmark_path):
                    os.unlink(landmark_path)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p>🔬 DeepFake Detection System</p>
        <p><strong>Team Members:</strong> Sena Yalçın, Emre Büyükyılmaz, İlayda Zeynep Karakaş</p>
        <p><strong>Guided by:</strong> Nazlı İkizler Cinbiş</p>
        <p><small>⚠️ This tool is for educational and research purposes. Results should be verified by experts for critical applications.</small></p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()