from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

st.set_page_config(page_title="AI Mood Chatbot", page_icon="🤖", layout="centered")

personalities = {
    "Funny": {
        "emoji": "😂",
        "prompt": "You are a funny chatbot who responds with humor",
        "color": "#f59e0b",
        "tagline": "Jokes, puns and good vibes",
    },
    "Angry": {
        "emoji": "😠",
        "prompt": "You are a angry chatbot who responds with irritated tone",
        "color": "#ef4444",
        "tagline": "Grumpy, impatient, but helpful",
    },
    "Sad": {
        "emoji": "😢",
        "prompt": "You are a sad chatbot who responds in a melancholic tone",
        "color": "#3b82f6",
        "tagline": "Every answer comes with a sigh",
    },
    "Romantic": {
        "emoji": "💖",
        "prompt": "You are a romantic chatbot who responds in a romantic tone",
        "color": "#ec4899",
        "tagline": "Poetic, sweet and dreamy",
    },
}


@st.cache_resource
def get_model():
    return ChatGoogleGenerativeAI(model="gemini-3-flash-preview")


def get_text(content):
    # Gemini may return a plain string or a list of content blocks
    if isinstance(content, str):
        return content
    return "".join(
        block.get("text", "") if isinstance(block, dict) else str(block)
        for block in content
    )


def reset_chat(personality):
    st.session_state.personality = personality
    st.session_state.messages = [SystemMessage(content=personalities[personality]["prompt"])]


# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("## 🤖 AI Chatbot")
    st.caption("Pick a personality and start chatting")

    choice = st.radio(
        "Personality",
        list(personalities),
        format_func=lambda p: f"{personalities[p]['emoji']}  {p}",
        label_visibility="collapsed",
    )

    st.divider()
    if st.button("🗑️  Clear chat", use_container_width=True):
        reset_chat(choice)
        st.rerun()

    st.caption("Powered by Gemini + LangChain")

# Start a fresh conversation on first load or when the personality changes
if "messages" not in st.session_state or st.session_state.personality != choice:
    reset_chat(choice)

p = personalities[choice]

# ---------- Styling ----------
st.markdown(
    f"""
    <style>
    .block-container {{ padding-top: 2rem; max-width: 820px; }}
    .hero {{
        background: linear-gradient(135deg, {p['color']}22, {p['color']}05);
        border: 1px solid {p['color']}55;
        border-radius: 18px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1.2rem;
        display: flex; align-items: center; gap: 1rem;
    }}
    .hero .emoji {{ font-size: 3rem; line-height: 1; }}
    .hero h1 {{ margin: 0; font-size: 1.7rem; }}
    .hero p {{ margin: 0.2rem 0 0; opacity: 0.75; }}
    [data-testid="stChatMessage"] {{
        border-radius: 14px;
        padding: 0.6rem 0.9rem;
        margin-bottom: 0.4rem;
    }}
    [data-testid="stChatInput"] textarea:focus {{ border-color: {p['color']}; }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="hero">
        <div class="emoji">{p['emoji']}</div>
        <div>
            <h1>{choice} Bot</h1>
            <p>{p['tagline']}</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------- Chat history ----------
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user", avatar="🧑"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant", avatar=p["emoji"]):
            st.markdown(msg.content)

if len(st.session_state.messages) == 1:
    st.info(f"Say hi to your {choice.lower()} chatbot {p['emoji']}")

# ---------- Input ----------
if prompt := st.chat_input("Type your message..."):
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar=p["emoji"]):
        with st.spinner("Thinking..."):
            try:
                response = get_model().invoke(st.session_state.messages)
                reply = get_text(response.content)
            except Exception as e:
                st.session_state.messages.pop()
                st.error(f"Something went wrong: {e}")
                st.stop()
        st.markdown(reply)

    st.session_state.messages.append(AIMessage(content=reply))
