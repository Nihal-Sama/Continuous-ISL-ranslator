import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode, RTCConfiguration
import av
import streamlit.components.v1 as components

from utils import extract_landmarks, draw_styled_landmarks, mediapipe_detection, mp_holistic
from translator import SmartTranslator

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Continuous-SignTranslator", page_icon="🌐", layout="centered")

# --- PREMIUM CUSTOM CSS ---
st.markdown("""
    <style>
    /* Hide all default Streamlit UI elements for a native app feel */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Sleek Application Header */
    .app-header {
        background: linear-gradient(135deg, #4A90E2 0%, #003366 100%);
        padding: 20px;
        border-radius: 20px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
        margin-bottom: 25px;
    }
    .app-title { font-size: 2.2rem; font-weight: 800; margin: 0; letter-spacing: 1px; }
    .app-subtitle { font-size: 1rem; opacity: 0.9; margin-top: 5px; font-weight: 300; }
    
    /* Modern Translation Output Card */
    .translation-card {
        background: #ffffff;
        border-radius: 24px;
        padding: 30px 20px;
        box-shadow: 0 15px 35px rgba(0,0,0,0.08);
        border: 1px solid #f0f0f0;
        text-align: center;
        margin-top: 20px;
        animation: fadeIn 0.5s ease-in-out;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .detected-label {
        font-size: 0.9rem;
        color: #95A5A6;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 600;
        margin-bottom: 10px;
    }
    
    .final-text {
        font-size: 2.8rem;
        font-weight: 800;
        color: #2C3E50;
        line-height: 1.2;
        margin-bottom: 15px;
    }
    
    .confidence-badge {
        display: inline-block;
        background: #E8F5E9;
        color: #2E7D32;
        padding: 6px 16px;
        border-radius: 30px;
        font-size: 0.85rem;
        font-weight: 700;
        box-shadow: 0 2px 5px rgba(46,125,50,0.1);
    }
    
    .error-card {
        background: #FDEDEC;
        border-radius: 15px;
        padding: 20px;
        color: #C0392B;
        text-align: center;
        font-weight: 600;
        border: 1px solid #FADBD8;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# --- LOAD AI MODELS ---
@st.cache_resource
def load_ai_models():
    model = keras.models.load_model("best_isl_model.keras")
    model.save("best_isl_model.h5")
    translator = SmartTranslator()
    return model, actions, translator

model, actions, translator = load_ai_models()

# --- TTS JAVASCRIPT ---
def speak_on_mobile(text):
    js_code = f"""
    <script>
        var msg = new SpeechSynthesisUtterance("{text}");
        msg.rate = 1.0;
        window.speechSynthesis.speak(msg);
    </script>
    """
    components.html(js_code, height=0)

# --- WEBRTC VIDEO PROCESSOR ---
class SignLanguageProcessor(VideoTransformerBase):
    def __init__(self):
        self.holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1)
        self.sequence = []
        self.is_recording = False

    def recv(self, frame):
        img = frame.to_ndarray(format="rgb24")
        
        img.flags.writeable = False
        results = self.holistic.process(img)
        img.flags.writeable = True
        
        draw_styled_landmarks(img, results)
        
        if self.is_recording:
            keypoints = extract_landmarks(results)
            self.sequence.append(keypoints)
            # Add a stylish recording indicator directly to the video feed
            cv2.circle(img, (50, 50), 20, (255, 0, 0), -1) 
            cv2.putText(img, "REC", (85, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3, cv2.LINE_AA)

        return av.VideoFrame.from_ndarray(img, format="rgb24")

# --- UI HEADER ---
st.markdown("""
<div class="app-header">
    <div class="app-title">🌐 SignTranslate Pro</div>
    <div class="app-subtitle">Real-Time Sign Language Interpreter</div>
</div>
""", unsafe_allow_html=True)

RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

ctx = webrtc_streamer(
    key="isl-translation",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=SignLanguageProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# --- DASHBOARD CONTROLS & OUTPUT ---
if ctx.video_processor:
    st.write("") # Spacer
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔴 Start Recording", use_container_width=True):
            ctx.video_processor.sequence = []
            ctx.video_processor.is_recording = True
            
    with col2:
        if st.button("✨ Translate Sign", type="primary", use_container_width=True):
            ctx.video_processor.is_recording = False
            recorded_data = ctx.video_processor.sequence
            
            if len(recorded_data) < 10:
                st.markdown('<div class="error-card">⚠️ Sign too short. Please try again.</div>', unsafe_allow_html=True)
            else:
                with st.spinner("Translating..."):
                    data = np.array(recorded_data)
                    sequence_length = 80
                    
                    if len(data) != sequence_length:
                        indices = np.linspace(0, len(data) - 1, sequence_length).astype(int)
                        data = data[indices]
                    
                    input_data = np.expand_dims(data, axis=0)
                    res = model.predict(input_data, verbose=0)[0]
                    confidence = res[np.argmax(res)]
                    
                    if confidence > 0.60:
                        raw_phrase = actions[np.argmax(res)].replace('_', ' ')
                        final_text = translator.enhance_sentence([raw_phrase])
                        
                        # --- PREMIUM TRANSLATION CARD ---
                        html_card = f"""
                        <div class="translation-card">
                            <div class="detected-label">Detected Action: {raw_phrase}</div>
                            <div class="final-text">"{final_text}"</div>
                            <div class="confidence-badge">Match Confidence: {confidence*100:.1f}%</div>
                        </div>
                        """
                        st.markdown(html_card, unsafe_allow_html=True)
                        speak_on_mobile(final_text)
                    else:

                        st.markdown(f'<div class="error-card">⚠️ Sign not recognized clearly ({confidence*100:.1f}%). Please try again.</div>', unsafe_allow_html=True)





