import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from tts import ttsservice
from gemini_chain import invoke_gemini
from llama_chain import invoke_llama
# Load multilingual pretrained model
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-multilingual-cased")
nlp = pipeline("text-classification", model=model, tokenizer=tokenizer)

# Function to process chatbot response
def chatbot_response(user_input):
    # Use NLP pipeline to classify input
    intent = nlp(user_input)
    label = intent[0]["label"]
    score = intent[0]["score"]

    # Example chatbot responses
    if label == "LABEL_0":
        return f"Эерэг утга илэрсэн байна! Итгэмжлэлийн түвшин: **{score:.2f}**."
    elif label == "LABEL_1":
        return f"Сөрөг утга илэрсэн байна. Итгэмжлэлийн түвшин: **{score:.2f}**."
    else:
        return "Уучлаарай, таны асуултыг ойлгосонгүй. Илүү тодорхой асуулт асуугаарай."

# Streamlit app layout
st.set_page_config(
    page_title="KHas Bank AI Chatbot",
    page_icon="💬",
    layout="wide",
)

# Custom CSS for Styling
st.markdown("""
    <style>
        body {
            background-color: #1e1e1e;
            color: #fff;
            font-family: 'Arial', sans-serif;
        }
        .stTextInput>div>div>input {
            background-color: #444;
            color: #fff;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 24px;
            font-size: 16px;
            border-radius: 5px;
            border: none;
            cursor: pointer;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
        .chat-message {
            margin-bottom: 15px;
            padding: 10px;
            border-radius: 8px;
            background-color: #333;
            max-width: 70%;
            margin-left: auto;
            margin-right: auto;
            color: white;
        }
        .user-message {
            background-color: #4CAF50;
        }
        .bot-message {
            background-color: #555;
        }
        .footer {
            text-align: center;
            font-size: 14px;
            color: #777;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar to show conversation history
if 'history' not in st.session_state:
    st.session_state['history'] = []

# Add conversation history to the sidebar
st.sidebar.title("💬 Chat History")
st.sidebar.write("Таны өмнөх чат:")
for message in st.session_state['history']:
    st.sidebar.markdown(f"- **{message['role']}**: {message['text']}")

# Sidebar
st.sidebar.title("💻 Chatbot Settings")
st.sidebar.write("Тохиргоо болон мэдээллүүд:")
st.sidebar.markdown(
    """
    - **Модел**: `bert-base-multilingual-cased`
    - **Тохируулсан**: Зээлийн асуулт, эерэг/сөрөг утга тодорхойлох.
    """
)

# Main UI
st.title("💬 KHas Bank AI Chatbot")
st.subheader("Танд ямар тусламж хэрэгтэй байна?")
st.write("Бид таны асуултад хариулахад бэлэн!")

# User input
with st.form(key="chat_form"):
    user_input = st.text_input("💬 Хэрэглэгч:", placeholder="Энд таны мессежийг бичнэ үү...")
    submit_button = st.form_submit_button(label="Асуух")

# Add conversation history and display chat
if submit_button and user_input:
    response = chatbot_response(user_input)
    if 'gemini' in user_input:
        user_input = user_input.replace('gemini', '')
        response = invoke_gemini(user_input)
    else:
        response =  invoke_llama(user_input)
        
    content = response['messages'][-1].content
    resp_audio = ttsservice(content)
    audio_html = f"""
    <audio hidden autoplay>
        <source src="data:audio/mp3;base64,{resp_audio}" type="audio/mp3">
        Your browser does not support the audio element.
    </audio>
    """
    st.markdown(audio_html, unsafe_allow_html=True)
    
    # Add user message and bot response to the history
    st.session_state['history'].append({'role': 'Хэрэглэгч', 'text': user_input})
    st.session_state['history'].append({'role': 'Бот', 'text': content})

    # Display chat messages on the main page
    st.markdown(f'<div class="chat-message user-message">💬 {user_input}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="chat-message bot-message">🤖 {content}</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
    <div class="footer">
        🤖 **AI Chatbot** © 2024. KHas Bankв .
    </div>
""", unsafe_allow_html=True)
