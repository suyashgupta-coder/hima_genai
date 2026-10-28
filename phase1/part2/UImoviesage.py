# MovieSage AI bot:
# 1 - Take raw para about a movie from user
# 2 - Extracts important structured information
# 3 - Generates a clean summary
# 4 - Returns it in a JSON format

from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from typing import List, Optional
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

st.set_page_config(page_title="MovieSage AI", page_icon="🎬", layout="wide")


# Pydantic schema
class Movie(BaseModel):
    title: str
    release_year: Optional[str] = None
    director: Optional[str] = None
    genre: List[str] = []
    main_cast: List[str] = []
    language: Optional[str] = None
    country: Optional[str] = None
    rating: Optional[str] = None
    runtime: Optional[str] = None
    plot_overview: str
    budget: Optional[str] = None
    box_office_collection: Optional[str] = None


parser = PydanticOutputParser(pydantic_object=Movie)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are MovieSage, an AI assistant specialized in analyzing movies.

Your task is to take raw, unstructured information about a movie and
transform it into structured JSON data.

Follow these rules:

1. Extract information only from the provided text.
2. Do not invent information.
3. If information is missing, use null or an empty list where appropriate.
4. Generate a clean and concise plot overview.
5. Return the result according to these formatting instructions:

{format_instructions}
"""
    ),
    (
        "human",
        """
Analyze the following raw movie information:

{paragraph}
"""
    )
])

SAMPLE = (
    "Inception is a 2010 science fiction action thriller written and directed by "
    "Christopher Nolan. The film stars Leonardo DiCaprio, Joseph Gordon-Levitt, Elliot Page, "
    "Tom Hardy and Marion Cotillard. Made in English as an American-British production, it runs "
    "for 148 minutes. DiCaprio plays Dom Cobb, a thief who steals secrets by entering people's "
    "dreams, and who is offered a chance at redemption if he can plant an idea in someone's mind "
    "instead. Made on a budget of around $160 million, the film earned over $836 million at the "
    "worldwide box office and holds a rating of 8.8/10 on IMDb."
)


@st.cache_resource
def get_model():
    return ChatGoogleGenerativeAI(model="gemini-3-flash-preview")


def analyze(paragraph):
    final_prompt = prompt.invoke({
        "paragraph": paragraph,
        "format_instructions": parser.get_format_instructions(),
    })
    res = get_model().invoke(final_prompt)
    # res.text works whether Gemini returns a string or a list of content blocks
    return parser.parse(res.text)


def load_sample():
    st.session_state.paragraph = SAMPLE


def show(value):
    return value if value else "—"


# ---------- Styling ----------
st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; max-width: 1200px; }
    .hero {
        background: linear-gradient(135deg, #7c3aed, #db2777);
        border-radius: 20px;
        padding: 1.8rem 2rem;
        color: white;
        margin-bottom: 1.5rem;
    }
    .hero h1 { margin: 0; font-size: 2.2rem; color: white; }
    .hero p { margin: 0.3rem 0 0; opacity: 0.9; }
    .movie-title { font-size: 2rem; font-weight: 700; margin-bottom: 0.2rem; }
    .chip {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        margin: 0 0.4rem 0.4rem 0;
        border-radius: 999px;
        background: rgba(124, 58, 237, 0.15);
        border: 1px solid rgba(124, 58, 237, 0.4);
        font-size: 0.85rem;
    }
    [data-testid="stMetric"] {
        background: rgba(128, 128, 128, 0.08);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 14px;
        padding: 0.8rem 1rem;
    }
    [data-testid="stMetricValue"] { font-size: 1.2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>🎬 MovieSage AI</h1>
        <p>Paste any paragraph about a movie and get clean, structured details in seconds.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

left, right = st.columns([2, 3], gap="large")

# ---------- Input ----------
with left:
    st.subheader("📝 Movie paragraph")
    paragraph = st.text_area(
        "Movie paragraph",
        key="paragraph",
        height=300,
        placeholder="e.g. Inception is a 2010 science fiction film directed by Christopher Nolan...",
        label_visibility="collapsed",
    )
    c1, c2 = st.columns(2)
    analyze_clicked = c1.button("✨ Analyze", type="primary", use_container_width=True)
    c2.button("🎲 Try a sample", on_click=load_sample, use_container_width=True)

    if analyze_clicked:
        if not paragraph.strip():
            st.warning("Please enter a paragraph about a movie first.")
        else:
            with st.spinner("MovieSage is reading your paragraph..."):
                try:
                    st.session_state.movie = analyze(paragraph)
                except Exception as e:
                    st.error(f"Something went wrong: {e}")

# ---------- Output ----------
with right:
    movie = st.session_state.get("movie")

    if movie is None:
        st.subheader("🎞️ Results")
        st.info("Your extracted movie details will appear here.")
    else:
        tab_card, tab_json = st.tabs(["🎞️ Movie card", "🧾 JSON"])

        with tab_card:
            year = f" ({movie.release_year})" if movie.release_year else ""
            st.markdown(f'<div class="movie-title">{movie.title}{year}</div>', unsafe_allow_html=True)
            if movie.director:
                st.caption(f"Directed by {movie.director}")

            if movie.genre:
                st.markdown(
                    "".join(f'<span class="chip">{g}</span>' for g in movie.genre),
                    unsafe_allow_html=True,
                )

            m1, m2, m3 = st.columns(3)
            m1.metric("⭐ Rating", show(movie.rating))
            m2.metric("⏱️ Runtime", show(movie.runtime))
            m3.metric("🗣️ Language", show(movie.language))

            m4, m5, m6 = st.columns(3)
            m4.metric("🌍 Country", show(movie.country))
            m5.metric("💰 Budget", show(movie.budget))
            m6.metric("🏆 Box office", show(movie.box_office_collection))

            st.markdown("#### 📖 Plot overview")
            st.write(movie.plot_overview)

            st.markdown("#### 🎭 Main cast")
            if movie.main_cast:
                st.markdown(
                    "".join(f'<span class="chip">{actor}</span>' for actor in movie.main_cast),
                    unsafe_allow_html=True,
                )
            else:
                st.caption("Not mentioned")

        with tab_json:
            movie_json = movie.model_dump_json(indent=2)
            st.code(movie_json, language="json")
            st.download_button(
                "⬇️ Download JSON",
                data=movie_json,
                file_name=f"{movie.title.replace(' ', '_').lower()}.json",
                mime="application/json",
                use_container_width=True,
            )
