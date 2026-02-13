import streamlit as st
import os
import json
import requests
import uuid
import base64
from io import BytesIO
from pypdf import PdfReader

# --- Configuration ---
GROQ_API_KEY = "Enter yours"
TEXT_MODEL = "llama-3.3-70b-versatile"
VISION_MODEL = "llama-3.2-11b-vision-preview"

st.set_page_config(
    page_title="MediAI Plus - Healthcare Assistant",
    page_icon="🏥",
    layout="wide"
)

# --- Premium High-Contrast CSS ---
st.markdown("""
<style>
    .stApp {
        background-color: #000a1a;
        background-image: 
            radial-gradient(at 0% 0%, rgba(0, 150, 255, 0.15) 0, transparent 50%), 
            radial-gradient(at 50% 0%, rgba(0, 255, 200, 0.1) 0, transparent 50%),
            linear-gradient(135deg, #001233 0%, #000a1a 100%);
        color: #ffffff;
    }
    [data-testid="stSidebar"] {
        background-color: rgba(0, 10, 26, 0.9) !important;
        backdrop-filter: blur(25px);
        border-right: 2px solid #00d2ff;
    }
    .stChatInputContainer {
        border-radius: 25px;
        background: #001a33 !important;
        border: 2px solid #00d2ff !important;
        box-shadow: 0 0 25px rgba(0, 210, 255, 0.5);
    }
    .stChatMessage {
        background: rgba(0, 30, 60, 0.7);
        border-radius: 20px;
        border: 1px solid #00d2ff;
        margin-bottom: 15px;
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.4);
    }
    h1, h2, h3 {
        color: #00d2ff !important;
        text-shadow: 0 0 25px rgba(0, 210, 255, 0.6);
        font-weight: 800;
    }
    .stButton>button {
        background: linear-gradient(90deg, #00d2ff, #0077ff);
        color: white;
        font-weight: 800;
        border-radius: 12px;
        border: none;
        text-transform: uppercase;
        box-shadow: 0 0 15px rgba(0, 210, 255, 0.4);
        transition: 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 0 30px rgba(0, 210, 255, 0.8);
    }
    .report-card {
        background: rgba(0, 26, 51, 0.9);
        border: 2px solid #00d2ff;
        padding: 20px;
        border-radius: 15px;
        margin-bottom: 20px;
    }
    .status-high { color: #ff4b4b; font-weight: bold; text-shadow: 0 0 10px rgba(255,75,75,0.5); }
    .status-medium { color: #f9d71c; font-weight: bold; text-shadow: 0 0 10px rgba(249,215,28,0.5); }
    .status-low { color: #00d2ff; font-weight: bold; text-shadow: 0 0 10px rgba(0,210,255,0.5); }
</style>
""", unsafe_allow_html=True)

# --- Utilities ---

def encode_image(image_file):
    return base64.b64encode(image_file.read()).decode('utf-8')

def extract_pdf_text(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def get_groq_response(messages, model, stream=True):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.2,
        "stream": stream
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, stream=stream)
        response.raise_for_status()
        
        if stream:
            for line in response.iter_lines():
                if line:
                    decoded = line.decode('utf-8').replace('data: ', '')
                    if decoded == '[DONE]': break
                    try:
                        content = json.loads(decoded)['choices'][0]['delta'].get('content', '')
                        if content: yield content
                    except: continue
        else:
            return response.json()['choices'][0]['message']['content']
    except Exception as e:
        yield f"⚠️ API Error: {str(e)}"

# --- UI Layout ---

st.title("🏥 MediAI Plus: Senior Health Suite")
st.markdown("### Advanced Multimodal Diagnosis & Lab Analysis")

# --- Sidebar Features ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3843/3843194.png", width=100)
    st.header("Intelligence Dashboard")
    
    # 1. Image Upload
    with st.expander("📸 Visual Symptom Checker"):
        image_file = st.file_uploader("Upload Symptom Photo", type=["jpg", "png", "jpeg"])
        if image_file:
            st.image(image_file, caption="Symptom Visual", use_container_width=True)

    # 2. PDF Lab Analyzer
    with st.expander("📄 Lab Report Analyzer", expanded=True):
        pdf_file = st.file_uploader("Upload Lab Report (PDF)", type=["pdf"])
        if pdf_file:
            st.success("PDF Loaded Successfully!")
            if st.button("� ANALYZE FULL REPORT", use_container_width=True):
                st.session_state.analyze_report = True

    # Health Tool
    with st.expander("⚖️ Quick BMI"):
        w = st.number_input("Weight (kg)", 10, 300, 70)
        h = st.number_input("Height (cm)", 50, 250, 175)
        if st.button("Calculate"):
            bmi = w / ((h/100)**2)
            st.info(f"BMI Score: {bmi:.1f}")

    st.divider()
    if st.button("🗑️ Reset All Cache", use_container_width=True):
        st.session_state.history = []
        st.session_state.analyze_report = False
        st.rerun()

# --- Analysis Logic ---

if "history" not in st.session_state:
    st.session_state.history = []
if "analyze_report" not in st.session_state:
    st.session_state.analyze_report = False

# Special Section for Automated Lab Analysis
if st.session_state.analyze_report and pdf_file:
    with st.status("Performing Comprehensive Lab Breakdown...", expanded=True) as status:
        st.write("Extracting clinical data...")
        raw_text = extract_pdf_text(pdf_file)
        
        st.write("Categorizing High/Low/Medium metrics...")
        analysis_prompt = f"""
        Analyze this laboratory report carefully. 
        Identify all medical markers (like Hemoglobin, WBC, Cholesterol, etc.) and categorize them exactly into:
        1. HIGH (Above normal range)
        2. MEDIUM (Within normal range)
        3. LOW (Below normal range)
        
        Provide a detailed summary statement explaining the most critical findings and what health actions should be considered.
        Format your response as a professional medical breakdown.
        
        REPORT TEXT:
        {raw_text[:3000]}
        """
        
        messages = [{"role": "system", "content": "You are a senior clinical pathologist AI. You specialize in categorizing lab results."},
                    {"role": "user", "content": analysis_prompt}]
        
        st.write("Synthesizing findings...")
        response_placeholder = st.empty()
        full_analysis = ""
        for chunk in get_groq_response(messages, TEXT_MODEL):
            full_analysis += chunk
            response_placeholder.markdown(full_analysis)
        
        st.session_state.history.append({"role": "assistant", "content": f"📋 **REPORT ANALYSIS COMPLETED**\n\n{full_analysis}"})
        status.update(label="Analysis Finished!", state="complete", expanded=False)
    st.session_state.analyze_report = False

# --- Chat Interface ---

# Display history
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Regular Interaction
if prompt := st.chat_input("Explain these symptoms or ask me anything..."):
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_res = ""
        
        # Prepare context if files exist
        context = ""
        if pdf_file:
            context += f"[PDF REPORT CONTEXT]: {extract_pdf_text(pdf_file)[:1500]}\n"
        
        messages = []
        if image_file:
            # Vision logic
            base64_img = encode_image(image_file)
            messages = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": f"{context}\n\nUser Question: {prompt}"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}
                ]
            }]
            model = VISION_MODEL
        else:
            # Text logic
            system_msg = "You are 'MediAI Plus', a high-contrast healthcare companion. Answer using professional clinical knowledge."
            messages = [{"role": "system", "content": system_msg}]
            for h in st.session_state.history[-5:]:
                messages.append(h)
            messages[-1]["content"] = f"{context}\n\n{prompt}"
            model = TEXT_MODEL
            
        for chunk in get_groq_response(messages, model):
            full_res += chunk
            placeholder.markdown(full_res + "▌")
        placeholder.markdown(full_res)
    
    st.session_state.history.append({"role": "assistant", "content": full_res})
