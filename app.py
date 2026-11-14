"""
ReviewSense - The Projection Room
=================================
A cinematic, editorial Streamlit web application for movie review sentiment classification.
Theme: "The Projection Room"
Palette:
    - Background: #0B0B0F
    - Surface: #14141B
    - Raised Surface: #1C1C26
    - Text: #F4EFE6
    - Muted: #9A97A3
    - Amber Accent (Projector Light): #FFB020
    - Emerald (Positive / Standing Ovation): #2EE59D
    - Crimson (Negative / Walkout): #FF3B5C
"""

import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

from preprocessing import clean_text

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="ReviewSense · Film Sentiment Classifier",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -----------------------------------------------------------------------------
# CSS INJECTION ("THE PROJECTION ROOM" DESIGN SYSTEM)
# -----------------------------------------------------------------------------
def inject_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:ital,opsz,wght@0,9..40,100..1000;1,9..40,100..1000&family=Fraunces:ital,opsz,wght@1,9..144,300;1,9..144,600&family=Space+Mono:ital,wght@0,400;0,700;1,400&display=swap');

        :root {
            --bg-base: #0B0B0F;
            --bg-surface: #14141B;
            --bg-raised: #1C1C26;
            --border-subtle: #272736;
            --border-highlight: #3D3D52;
            --text-main: #F4EFE6;
            --text-muted: #9A97A3;
            --amber-accent: #FFB020;
            --amber-glow: rgba(255, 176, 32, 0.22);
            --emerald-pos: #2EE59D;
            --emerald-glow: rgba(46, 229, 157, 0.25);
            --crimson-neg: #FF3B5C;
            --crimson-glow: rgba(255, 59, 92, 0.25);
            --font-display: 'Bebas Neue', sans-serif;
            --font-editorial: 'Fraunces', serif;
            --font-body: 'DM Sans', sans-serif;
            --font-mono: 'Space Mono', monospace;
        }

        /* Hide default Streamlit chrome */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        .stDeployButton {display: none;}
        div[data-testid="stToolbar"] {display: none;}

        /* Main application container styling */
        .stApp {
            background-color: var(--bg-base);
            color: var(--text-main);
            font-family: var(--font-body);
            background-image: 
                radial-gradient(circle at 50% -10%, rgba(255, 176, 32, 0.12) 0%, transparent 65%),
                linear-gradient(180deg, rgba(11, 11, 15, 0.8) 0%, rgba(11, 11, 15, 1) 100%);
            animation: lightsDim 0.9s cubic-bezier(0.16, 1, 0.3, 1);
        }

        @keyframes lightsDim {
            0% { opacity: 0; filter: brightness(0.65) blur(1px); }
            100% { opacity: 1; filter: brightness(1) blur(0); }
        }

        /* Sprocket holes on section dividers */
        .sprocket-border {
            height: 12px;
            width: 100%;
            background: repeating-linear-gradient(
                90deg,
                #1C1C26 0px,
                #1C1C26 12px,
                transparent 12px,
                transparent 22px
            );
            border-radius: 2px;
            margin: 22px 0;
            opacity: 0.85;
        }

        /* Hero Banner */
        .hero-container {
            text-align: center;
            padding: 30px 10px 15px 10px;
            position: relative;
        }

        .hero-title {
            font-family: var(--font-display);
            font-size: clamp(3.5rem, 8vw, 6.5rem);
            letter-spacing: 0.06em;
            color: var(--text-main);
            margin: 0;
            line-height: 0.95;
            text-transform: uppercase;
            text-shadow: 0 4px 28px rgba(255, 176, 32, 0.28);
        }

        .hero-subtitle {
            font-family: var(--font-editorial);
            font-style: italic;
            font-size: clamp(1.15rem, 2.5vw, 1.7rem);
            color: var(--amber-accent);
            margin-top: 10px;
            font-weight: 300;
            letter-spacing: 0.02em;
        }

        /* Slow Marquee Ticker */
        .marquee-container {
            width: 100%;
            overflow: hidden;
            background: #14141B;
            border-top: 1px solid var(--border-subtle);
            border-bottom: 1px solid var(--border-subtle);
            padding: 9px 0;
            margin: 18px 0 25px 0;
            white-space: nowrap;
        }

        .marquee-content {
            display: inline-block;
            font-family: var(--font-mono);
            font-size: 0.82rem;
            color: var(--amber-accent);
            letter-spacing: 0.18em;
            text-transform: uppercase;
            animation: marqueeScroll 32s linear infinite;
        }

        @keyframes marqueeScroll {
            0% { transform: translateX(0%); }
            100% { transform: translateX(-50%); }
        }

        /* Ticket Stub Metric Cards */
        .ticket-stub {
            background: var(--bg-surface);
            border: 1px solid var(--border-subtle);
            border-radius: 8px;
            padding: 20px 22px;
            position: relative;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
            transition: transform 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease;
            margin-bottom: 15px;
            overflow: hidden;
        }

        .ticket-stub:hover {
            transform: translateY(-4px);
            border-color: var(--amber-accent);
            box-shadow: 0 12px 30px var(--amber-glow);
        }

        .ticket-serial {
            font-family: var(--font-mono);
            font-size: 0.72rem;
            color: var(--text-muted);
            letter-spacing: 0.14em;
            text-transform: uppercase;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px dashed var(--border-highlight);
            padding-bottom: 8px;
            margin-bottom: 12px;
        }

        .ticket-title {
            font-family: var(--font-display);
            font-size: 1.35rem;
            letter-spacing: 0.06em;
            color: var(--text-muted);
            margin: 0;
            text-transform: uppercase;
        }

        .ticket-value {
            font-family: var(--font-mono);
            font-size: 2.2rem;
            font-weight: 700;
            color: var(--text-main);
            margin: 6px 0 2px 0;
            letter-spacing: -0.02em;
        }

        .ticket-desc {
            font-size: 0.82rem;
            color: var(--text-muted);
            margin: 0;
        }

        /* Film Frame Cards */
        .film-frame {
            background: var(--bg-surface);
            border: 1px solid var(--border-subtle);
            border-radius: 8px;
            padding: 20px 18px;
            height: 100%;
            transition: transform 0.2s ease, border-color 0.2s ease;
        }

        .film-frame:hover {
            transform: translateY(-3px);
            border-color: var(--amber-accent);
        }

        .frame-number {
            font-family: var(--font-mono);
            font-size: 0.75rem;
            color: var(--amber-accent);
            letter-spacing: 0.12em;
            margin-bottom: 6px;
        }

        .frame-title {
            font-family: var(--font-display);
            font-size: 1.45rem;
            letter-spacing: 0.05em;
            color: var(--text-main);
            margin: 0 0 8px 0;
        }

        .frame-desc {
            font-size: 0.88rem;
            color: var(--text-muted);
            line-height: 1.45;
            margin: 0;
        }

        /* Verdict Cards */
        .verdict-box {
            border-radius: 12px;
            padding: 30px;
            margin: 25px 0;
            text-align: center;
            position: relative;
            background: var(--bg-surface);
            transition: all 0.3s ease;
        }

        .verdict-positive {
            border: 2px solid var(--emerald-pos);
            box-shadow: 0 0 35px var(--emerald-glow), inset 0 0 25px rgba(46, 229, 157, 0.06);
        }

        .verdict-negative {
            border: 2px solid var(--crimson-neg);
            box-shadow: 0 0 35px var(--crimson-glow), inset 0 0 25px rgba(255, 59, 92, 0.06);
        }

        .verdict-badge {
            font-family: var(--font-mono);
            font-size: 0.85rem;
            font-weight: 700;
            letter-spacing: 0.2em;
            text-transform: uppercase;
            display: inline-block;
            padding: 6px 16px;
            border-radius: 30px;
            margin-bottom: 14px;
        }

        .badge-positive {
            background: rgba(46, 229, 157, 0.15);
            color: var(--emerald-pos);
            border: 1px solid var(--emerald-pos);
        }

        .badge-negative {
            background: rgba(255, 59, 92, 0.15);
            color: var(--crimson-neg);
            border: 1px solid var(--crimson-neg);
        }

        .verdict-title {
            font-family: var(--font-display);
            font-size: clamp(3rem, 6vw, 4.8rem);
            margin: 0;
            letter-spacing: 0.08em;
            line-height: 1;
        }

        .title-positive {
            color: var(--emerald-pos);
            text-shadow: 0 2px 20px rgba(46, 229, 157, 0.4);
        }

        .title-negative {
            color: var(--crimson-neg);
            text-shadow: 0 2px 20px rgba(255, 59, 92, 0.4);
        }

        .confidence-label {
            font-family: var(--font-mono);
            font-size: 1.05rem;
            color: var(--text-main);
            margin-top: 14px;
            letter-spacing: 0.05em;
        }

        /* Progress Bar Custom */
        .conf-progress-track {
            width: 100%;
            height: 12px;
            background: #1C1C26;
            border-radius: 6px;
            overflow: hidden;
            margin: 16px auto 10px auto;
            max-width: 520px;
            border: 1px solid var(--border-subtle);
        }

        .conf-progress-fill {
            height: 100%;
            border-radius: 6px;
            transition: width 0.8s ease-in-out;
        }

        /* Echoed Review Block */
        .echo-quote {
            background: var(--bg-surface);
            border-left: 4px solid var(--amber-accent);
            padding: 16px 20px;
            border-radius: 0 8px 8px 0;
            margin: 18px 0;
            font-family: var(--font-editorial);
            font-style: italic;
            font-size: 1.08rem;
            color: var(--text-main);
            line-height: 1.5;
        }

        /* Editorial Info Cards (About tab) */
        .info-card {
            background: var(--bg-surface);
            border: 1px solid var(--border-subtle);
            border-radius: 8px;
            padding: 24px;
            margin-bottom: 20px;
            transition: border-color 0.25s ease;
        }

        .info-card:hover {
            border-color: var(--amber-accent);
        }

        .info-header {
            font-family: var(--font-display);
            font-size: 1.6rem;
            letter-spacing: 0.05em;
            color: var(--amber-accent);
            margin: 0 0 10px 0;
        }

        .info-body {
            font-size: 0.94rem;
            color: var(--text-main);
            line-height: 1.6;
        }

        /* Streamlit Tab Overrides */
        .stTabs [data-baseweb="tab-list"] {
            gap: 12px;
            background-color: transparent;
            border-bottom: 1px solid var(--border-subtle);
            padding-bottom: 6px;
        }

        .stTabs [data-baseweb="tab"] {
            font-family: var(--font-display);
            font-size: 1.35rem;
            letter-spacing: 0.08em;
            color: var(--text-muted);
            background: transparent;
            border: 1px solid transparent;
            border-radius: 6px;
            padding: 8px 22px;
            transition: all 0.2s ease;
        }

        .stTabs [data-baseweb="tab"]:hover {
            color: var(--text-main);
            background: var(--bg-surface);
        }

        .stTabs [aria-selected="true"] {
            color: var(--amber-accent) !important;
            background: var(--bg-surface) !important;
            border: 1px solid var(--amber-accent) !important;
            box-shadow: 0 0 16px var(--amber-glow);
        }

        /* Input and Button Overrides */
        div[data-baseweb="textarea"] {
            background-color: var(--bg-surface) !important;
            border-color: var(--border-subtle) !important;
            border-radius: 8px !important;
        }

        div[data-baseweb="textarea"]:focus-within {
            border-color: var(--amber-accent) !important;
            box-shadow: 0 0 14px var(--amber-glow) !important;
        }

        textarea {
            color: var(--text-main) !important;
            font-family: var(--font-body) !important;
            font-size: 1rem !important;
        }

        .stButton button {
            border-radius: 6px !important;
            font-family: var(--font-mono) !important;
            font-size: 0.88rem !important;
            letter-spacing: 0.06em !important;
            text-transform: uppercase !important;
            padding: 10px 24px !important;
            transition: all 0.25s ease !important;
        }

        .stButton button[kind="primary"] {
            background: var(--amber-accent) !important;
            color: #0B0B0F !important;
            border: none !important;
            font-weight: 700 !important;
            box-shadow: 0 4px 18px var(--amber-glow) !important;
        }

        .stButton button[kind="primary"]:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 24px rgba(255, 176, 32, 0.45) !important;
        }

        .stButton button[kind="secondary"] {
            background: var(--bg-surface) !important;
            color: var(--text-main) !important;
            border: 1px solid var(--border-subtle) !important;
        }

        .stButton button[kind="secondary"]:hover {
            border-color: var(--amber-accent) !important;
            color: var(--amber-accent) !important;
        }

        /* "BEST CUT" Badge */
        .best-cut-badge {
            display: inline-block;
            background: rgba(255, 176, 32, 0.15);
            border: 1px solid var(--amber-accent);
            color: var(--amber-accent);
            font-family: var(--font-mono);
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.14em;
            padding: 4px 10px;
            border-radius: 4px;
            margin-left: 8px;
            vertical-align: middle;
        }

        /* Responsive tweaks */
        @media (max-width: 600px) {
            .ticket-value { font-size: 1.8rem; }
            .hero-title { font-size: 3.2rem; }
            .verdict-box { padding: 20px 14px; }
            .verdict-title { font-size: 2.6rem; }
            .stTabs [data-baseweb="tab"] { font-size: 1.05rem; padding: 6px 12px; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# -----------------------------------------------------------------------------
# DATA & MODEL LOADERS (CACHED)
# -----------------------------------------------------------------------------
@st.cache_resource
def load_model_and_vectorizer():
    model_path = os.path.join("models", "best_sentiment_model.pkl")
    vectorizer_path = os.path.join("models", "tfidf_vectorizer.pkl")
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        return None, None
    try:
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        return model, vectorizer
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None


@st.cache_data
def load_metrics():
    metrics_path = os.path.join("models", "metrics.json")
    if not os.path.exists(metrics_path):
        return None
    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error reading metrics telemetry: {e}")
        return None


# -----------------------------------------------------------------------------
# APPLICATION ENTRYPOINT
# -----------------------------------------------------------------------------
def main():
    inject_css()

    model, vectorizer = load_model_and_vectorizer()
    metrics = load_metrics()

    # Guardrail: Ensure models and metrics exist
    if model is None or vectorizer is None or metrics is None:
        st.markdown(
            """
            <div style="background:#14141B; border: 2px solid #FF3B5C; border-radius: 12px; padding: 40px; text-align: center; margin-top: 60px;">
                <h1 style="font-family:'Bebas Neue', sans-serif; font-size: 3rem; color: #FF3B5C; letter-spacing: 0.05em; margin: 0;">
                    PROJECTION ROOM OFFLINE · MODEL ARTIFACTS MISSING
                </h1>
                <p style="font-family:'DM Sans', sans-serif; font-size: 1.15rem; color: #F4EFE6; margin-top: 15px;">
                    ReviewSense requires trained model artifacts and vectorizer parameters to screen reviews.
                </p>
                <div style="background:#1C1C26; border-radius: 8px; padding: 18px; margin: 25px auto; max-width: 650px; font-family:'Space Mono', monospace; font-size: 0.95rem; color: #FFB020; border: 1px dashed #3D3D52;">
                    ▶ Run the training pipeline in your terminal:<br>
                    <strong>python train.py</strong>
                </div>
                <p style="font-size: 0.9rem; color: #9A97A3;">
                    This will inspect <code>dataset/IMDB Dataset.csv</code>, train Logistic Regression & Naive Bayes, and export all necessary artifacts to <code>models/</code>.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # Extract dynamic dataset metrics
    dataset_stats = metrics.get("dataset_stats", {})
    total_reviews = dataset_stats.get("total_clean", 49582)
    pos_reviews = dataset_stats.get("positive_count", 24884)
    neg_reviews = dataset_stats.get("negative_count", 24698)
    best_model_info = metrics.get("best_model", {})
    best_model_name = best_model_info.get("name", "LogisticRegression")
    best_f1 = best_model_info.get("f1_score", 0.9032) * 100

    # -------------------------------------------------------------------------
    # HERO HEADER & MARQUEE TICKER
    # -------------------------------------------------------------------------
    st.markdown(
        f"""
        <div class="hero-container">
            <h1 class="hero-title">REVIEWSENSE</h1>
            <div class="hero-subtitle">Every review has a verdict.</div>
        </div>
        <div class="marquee-container">
            <div class="marquee-content">
                NOW SCREENING · {total_reviews:,} REVIEWS · 2 MODELS EVALUATED · 1 CHAMPION VERDICT ({best_model_name} {best_f1:.1f}% F1) · STRATIFIED 80/20 SPLIT · TF-IDF (1,2) N-GRAMS · PRODUCTION READY ·
                NOW SCREENING · {total_reviews:,} REVIEWS · 2 MODELS EVALUATED · 1 CHAMPION VERDICT ({best_model_name} {best_f1:.1f}% F1) · STRATIFIED 80/20 SPLIT · TF-IDF (1,2) N-GRAMS · PRODUCTION READY ·
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -------------------------------------------------------------------------
    # MAIN NAVIGATION TABS
    # -------------------------------------------------------------------------
    tab_home, tab_analyze, tab_perf, tab_about = st.tabs([
        "🎞️  HOME",
        "🎯  ANALYZE",
        "📊  PERFORMANCE",
        "📖  ABOUT"
    ])

    # =========================================================================
    # TAB 1: HOME
    # =========================================================================
    with tab_home:
        # Ticket-stub metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(
                f"""
                <div class="ticket-stub">
                    <div class="ticket-serial">
                        <span>SERIES 2026</span>
                        <span>NO. RS-000482</span>
                    </div>
                    <div class="ticket-title">Total Film Archive</div>
                    <div class="ticket-value">{total_reviews:,}</div>
                    <p class="ticket-desc">Deduplicated IMDb reviews analyzed</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                f"""
                <div class="ticket-stub">
                    <div class="ticket-serial">
                        <span style="color:var(--emerald-pos);">CRITICAL ACCLAIM</span>
                        <span>NO. RS-000483</span>
                    </div>
                    <div class="ticket-title">Standing Ovations</div>
                    <div class="ticket-value" style="color:var(--emerald-pos);">{pos_reviews:,}</div>
                    <p class="ticket-desc">{dataset_stats.get('positive_pct', 50.19)}% Positive reviews in dataset</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col3:
            st.markdown(
                f"""
                <div class="ticket-stub">
                    <div class="ticket-serial">
                        <span style="color:var(--crimson-neg);">BOX OFFICE FLOPS</span>
                        <span>NO. RS-000484</span>
                    </div>
                    <div class="ticket-title">Walkouts & Pans</div>
                    <div class="ticket-value" style="color:var(--crimson-neg);">{neg_reviews:,}</div>
                    <p class="ticket-desc">{dataset_stats.get('negative_pct', 49.81)}% Negative reviews in dataset</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown('<div class="sprocket-border"></div>', unsafe_allow_html=True)

        # Dataset Class Distribution Chart & Editorial Overview
        ch_col1, ch_col2 = st.columns([1.1, 1])

        with ch_col1:
            st.markdown(
                """
                <h3 style="font-family:'Bebas Neue', sans-serif; font-size:1.8rem; letter-spacing:0.05em; color:var(--text-main); margin-bottom: 5px;">
                    CORPUS BALANCE & INTEGRITY
                </h3>
                <p style="font-size:0.92rem; color:var(--text-muted); line-height:1.5;">
                    The IMDb sentiment corpus is balanced naturally across 50,000 international spectator reviews. 
                    Prior to train/test partitioning, 418 duplicate entries were identified and dropped to guarantee 
                    zero train-test data contamination.
                </p>
                """,
                unsafe_allow_html=True,
            )

            # Donut chart with dark theme
            fig, ax = plt.subplots(figsize=(6, 3.6), facecolor="#0B0B0F")
            ax.set_facecolor("#0B0B0F")
            colors = ["#2EE59D", "#FF3B5C"]
            sizes = [pos_reviews, neg_reviews]
            labels = ["Positive (50.19%)", "Negative (49.81%)"]
            
            pie_result = ax.pie(
                sizes,
                labels=labels,
                autopct="%1.1f%%",
                startangle=140,
                colors=colors,
                pctdistance=0.76,
                textprops={"color": "#F4EFE6", "fontsize": 10, "fontfamily": "sans-serif"},
                wedgeprops={"width": 0.42, "edgecolor": "#14141B", "linewidth": 2}
            )
            if len(pie_result) == 3:
                autotexts = pie_result[2]
                for autotext in autotexts:
                    autotext.set_color("#0B0B0F")
                    autotext.set_fontweight("bold")
                    autotext.set_fontsize(11)

            # Center text in donut
            ax.text(0, 0, f"{total_reviews:,}\nREVIEWS", ha="center", va="center",
                    color="#FFB020", fontsize=11, fontweight="bold", fontfamily="monospace")

            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with ch_col2:
            st.markdown(
                f"""
                <h3 style="font-family:'Bebas Neue', sans-serif; font-size:1.8rem; letter-spacing:0.05em; color:var(--text-main); margin-bottom: 5px;">
                    SCREENING PROTOCOL
                </h3>
                <p style="font-size:0.92rem; color:var(--text-muted); line-height:1.5; margin-bottom: 16px;">
                    ReviewSense operates on a rigorous, leak-free machine learning architecture designed to mirror commercial grade NLP systems.
                </p>
                <div style="background:var(--bg-surface); border:1px solid var(--border-subtle); border-radius:8px; padding:16px; margin-bottom:12px;">
                    <div style="font-family:'Space Mono', monospace; font-size:0.8rem; color:var(--amber-accent);">01 · STRATIFIED SAMPLING</div>
                    <div style="font-size:0.9rem; color:var(--text-main); margin-top:4px;">
                        Held-out 20% test slice (9,917 reviews) preserved strictly for final evaluation with exact class distribution mirroring.
                    </div>
                </div>
                <div style="background:var(--bg-surface); border:1px solid var(--border-subtle); border-radius:8px; padding:16px; margin-bottom:12px;">
                    <div style="font-family:'Space Mono', monospace; font-size:0.8rem; color:var(--amber-accent);">02 · NEGATION RESCUE</div>
                    <div style="font-size:0.9rem; color:var(--text-main); margin-top:4px;">
                        Standard stop-word lists inadvertently erase words like <em>not</em> and <em>never</em>. ReviewSense filters stop-words while vigilantly protecting 20+ polarity qualifiers.
                    </div>
                </div>
                <div style="background:var(--bg-surface); border:1px solid var(--border-subtle); border-radius:8px; padding:16px;">
                    <div style="font-family:'Space Mono', monospace; font-size:0.8rem; color:var(--amber-accent);">03 · CHAMPION CUT</div>
                    <div style="font-size:0.9rem; color:var(--text-main); margin-top:4px;">
                        <strong>{best_model_name}</strong> demonstrated peak discriminative accuracy ({metrics['models'][best_model_name]['accuracy']*100:.2f}%) and balanced F1-score ({best_f1:.2f}%).
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown('<div class="sprocket-border"></div>', unsafe_allow_html=True)

        # 4-Step Film Frame Process Strip
        st.markdown(
            """
            <h3 style="font-family:'Bebas Neue', sans-serif; font-size:2rem; letter-spacing:0.06em; color:var(--text-main); text-align:center; margin-bottom: 20px;">
                THE PROJECTION PIPELINE · 4 REEL TRANSFORMATION
            </h3>
            """,
            unsafe_allow_html=True,
        )

        f_col1, f_col2, f_col3, f_col4 = st.columns(4)
        with f_col1:
            st.markdown(
                """
                <div class="film-frame">
                    <div class="frame-number">REEL 01</div>
                    <div class="frame-title">RAW REEL</div>
                    <p class="frame-desc">
                        Spectator enters authentic movie critique containing informal syntax, punctuation, and embedded HTML markup.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with f_col2:
            st.markdown(
                """
                <div class="film-frame">
                    <div class="frame-number">REEL 02</div>
                    <div class="frame-title">CLEAN CUT</div>
                    <p class="frame-desc">
                        Shared <code>clean_text()</code> pipeline strips HTML tags, expands contractions, removes noise, and preserves negations.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with f_col3:
            st.markdown(
                """
                <div class="film-frame">
                    <div class="frame-number">REEL 03</div>
                    <div class="frame-title">TF-IDF LENS</div>
                    <p class="frame-desc">
                        Transforms cleaned text into 50,000 sublinear n-gram features (unigrams & bigrams) using the train-fitted vectorizer.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with f_col4:
            st.markdown(
                """
                <div class="film-frame">
                    <div class="frame-number">REEL 04</div>
                    <div class="frame-title">THE VERDICT</div>
                    <p class="frame-desc">
                        Logistic regression classifier computes calibrated probability score: <em>Standing Ovation</em> vs. <em>Walkout</em>.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # =========================================================================
    # TAB 2: ANALYZE (INTERACTIVE CLASSIFIER)
    # =========================================================================
    with tab_analyze:
        st.markdown(
            """
            <div style="margin-bottom: 15px;">
                <h2 style="font-family:'Bebas Neue', sans-serif; font-size:2.2rem; letter-spacing:0.05em; color:var(--text-main); margin-bottom: 4px;">
                    SENTIMENT SCREENING ROOM
                </h2>
                <p style="font-size:0.95rem; color:var(--text-muted); margin:0;">
                    Type or paste a movie review below, or audition with one of the cinematic presets.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Initialize session state for review text
        if "review_text" not in st.session_state:
            st.session_state.review_text = ""

        # Preset test cases
        st.markdown(
            """<span style="font-family:'Space Mono', monospace; font-size:0.75rem; color:var(--amber-accent); text-transform:uppercase; letter-spacing:0.12em;">Load Screening Presets:</span>""",
            unsafe_allow_html=True
        )

        p_col1, p_col2, p_col3 = st.columns(3)
        with p_col1:
            if st.button("🌟 Positive Masterpiece", use_container_width=True, type="secondary"):
                st.session_state.review_text = (
                    "A timeless cinematic masterpiece with breathtaking visuals, flawless direction, and deeply moving performances throughout."
                )
        with p_col2:
            if st.button("💀 Disastrous Walkout", use_container_width=True, type="secondary"):
                st.session_state.review_text = (
                    "An utter catastrophe and total waste of time. The dialogue was laughably bad, pacing was dreadful, and the acting was completely wooden."
                )
        with p_col3:
            if st.button("🎭 Subtle Negation Test", use_container_width=True, type="secondary"):
                st.session_state.review_text = (
                    "The movie was not good at all, despite what the dazzling promotional trailers claimed."
                )

        # Review Text Area
        user_input = st.text_area(
            label="Movie Review Transcript",
            value=st.session_state.review_text,
            height=145,
            placeholder="e.g. 'Denis Villeneuve crafts an intoxicating sci-fi symphony with stunning sound design and hypnotic pacing...'",
            label_visibility="collapsed",
            key="input_area"
        )

        # Action Buttons
        btn_col1, btn_col2, _ = st.columns([1.2, 0.9, 3])
        with btn_col1:
            predict_clicked = st.button("🎬  PREDICT SENTIMENT", type="primary", use_container_width=True)
        with btn_col2:
            if st.button("↺  CLEAR", type="secondary", use_container_width=True):
                st.session_state.review_text = ""
                st.rerun()

        # Handle prediction
        if predict_clicked:
            raw_text = user_input.strip()

            # Guardrail: Check for empty or whitespace-only review
            if not raw_text:
                st.markdown(
                    """
                    <div style="background:#14141B; border: 1px solid #FFB020; border-radius: 8px; padding: 18px; margin: 20px 0; display:flex; align-items:center; gap: 14px;">
                        <span style="font-size:1.8rem;">⚠️</span>
                        <div>
                            <div style="font-family:'Space Mono', monospace; font-size:0.85rem; color:#FFB020; font-weight:bold;">BLANK FILM STRIP</div>
                            <div style="font-size:0.92rem; color:#F4EFE6;">The projector cannot roll an empty frame. Please type or paste a review before requesting a verdict.</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                # Run through shared clean_text pipeline
                cleaned = clean_text(raw_text)

                # Guardrail: Check if cleaning stripped everything (e.g. only symbols or punctuation)
                if not cleaned:
                    st.markdown(
                        """
                        <div style="background:#14141B; border: 1px solid #FFB020; border-radius: 8px; padding: 18px; margin: 20px 0; display:flex; align-items:center; gap: 14px;">
                            <span style="font-size:1.8rem;">⚠️</span>
                            <div>
                                <div style="font-family:'Space Mono', monospace; font-size:0.85rem; color:#FFB020; font-weight:bold;">UNINTELLIGIBLE AUDIO REEL</div>
                                <div style="font-size:0.92rem; color:#F4EFE6;">The input contained only symbols or uninformative noise. No valid lexical tokens were found.</div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    # Feature transformation & model inference
                    feat_vec = vectorizer.transform([cleaned])
                    pred = int(model.predict(feat_vec)[0])
                    probabilities = model.predict_proba(feat_vec)[0]
                    prob_neg = float(probabilities[0])
                    prob_pos = float(probabilities[1])

                    is_positive = (pred == 1)
                    confidence_pct = (prob_pos if is_positive else prob_neg) * 100

                    # Echoed Review Quote Block
                    st.markdown(
                        f"""
                        <div class="echo-quote">
                            "{raw_text}"
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # Dynamic Verdict Card with Glow
                    box_class = "verdict-positive" if is_positive else "verdict-negative"
                    badge_class = "badge-positive" if is_positive else "badge-negative"
                    title_class = "title-positive" if is_positive else "title-negative"
                    verdict_word = "POSITIVE" if is_positive else "NEGATIVE"
                    sub_verdict = "STANDING OVATION" if is_positive else "WALKOUT"
                    fill_color = "var(--emerald-pos)" if is_positive else "var(--crimson-neg)"

                    st.markdown(
                        f"""
                        <div class="verdict-box {box_class}">
                            <div class="verdict-badge {badge_class}">{sub_verdict}</div>
                            <h2 class="verdict-title {title_class}">{verdict_word}</h2>
                            <div class="confidence-label">
                                CONFIDENCE: <strong>{confidence_pct:.1f}%</strong>
                            </div>
                            <div class="conf-progress-track">
                                <div class="conf-progress-fill" style="width: {confidence_pct:.1f}%; background: {fill_color};"></div>
                            </div>
                            <div style="font-family:'Space Mono', monospace; font-size:0.82rem; color:var(--text-muted); margin-top:8px;">
                                Acclaim Probability: <span style="color:var(--emerald-pos);">{prob_pos*100:.1f}%</span> &nbsp;|&nbsp; 
                                Disapproval Probability: <span style="color:var(--crimson-neg);">{prob_neg*100:.1f}%</span>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # Pipeline Transparency Expander
                    with st.expander("🔍 Inspect Internal Pipeline & Token Breakdown"):
                        exp_col1, exp_col2 = st.columns(2)
                        with exp_col1:
                            st.markdown(
                                f"""
                                <strong style="color:var(--amber-accent); font-family:'Space Mono', monospace;">Raw Input Length:</strong> {len(raw_text)} chars<br>
                                <strong style="color:var(--amber-accent); font-family:'Space Mono', monospace;">Normalized Text:</strong><br>
                                <code>{cleaned}</code>
                                """,
                                unsafe_allow_html=True
                            )
                        with exp_col2:
                            st.markdown(
                                f"""
                                <strong style="color:var(--amber-accent); font-family:'Space Mono', monospace;">Active Model:</strong> {best_model_name}<br>
                                <strong style="color:var(--amber-accent); font-family:'Space Mono', monospace;">Non-Zero TF-IDF Features:</strong> {feat_vec.nnz:,} tokens<br>
                                <strong style="color:var(--amber-accent); font-family:'Space Mono', monospace;">Decision Boundary Margin:</strong> {prob_pos - prob_neg:+.4f}
                                """,
                                unsafe_allow_html=True
                            )

    # =========================================================================
    # TAB 3: PERFORMANCE (MODEL LEADERBOARD & METRICS)
    # =========================================================================
    with tab_perf:
        st.markdown(
            """
            <div style="margin-bottom: 20px;">
                <h2 style="font-family:'Bebas Neue', sans-serif; font-size:2.2rem; letter-spacing:0.05em; color:var(--text-main); margin-bottom: 4px;">
                    MODEL EVALUATION & BENCHMARKS
                </h2>
                <p style="font-size:0.95rem; color:var(--text-muted); margin:0;">
                    Rigorous benchmarking conducted strictly on the 20% held-out test split (9,917 reviews). Zero data leakage.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        models_data = metrics.get("models", {})

        # Model Comparison Table
        table_rows = []
        for m_name, m_stats in models_data.items():
            is_best = (m_name == best_model_name)
            name_display = f"{m_name} ⭐ BEST CUT" if is_best else m_name
            table_rows.append({
                "Model": name_display,
                "Accuracy": f"{m_stats['accuracy']*100:.2f}%",
                "Precision": f"{m_stats['precision']*100:.2f}%",
                "Recall": f"{m_stats['recall']*100:.2f}%",
                "F1-Score": f"{m_stats['f1_score']*100:.2f}%",
                "Training Time": f"{m_stats['training_time_sec']:.2f}s"
            })

        df_comparison = pd.DataFrame(table_rows)
        st.dataframe(df_comparison, use_container_width=True, hide_index=True)

        st.markdown('<div class="sprocket-border"></div>', unsafe_allow_html=True)

        # Grouped Bar Chart of Performance Metrics
        st.markdown(
            """
            <h3 style="font-family:'Bebas Neue', sans-serif; font-size:1.8rem; letter-spacing:0.05em; color:var(--text-main); margin-bottom: 15px;">
                CROSS-MODEL METRIC COMPARISON
            </h3>
            """,
            unsafe_allow_html=True
        )

        metric_names = ["Accuracy", "Precision", "Recall", "F1-Score"]
        lr_scores = [
            models_data.get("LogisticRegression", {}).get("accuracy", 0.9018) * 100,
            models_data.get("LogisticRegression", {}).get("precision", 0.8935) * 100,
            models_data.get("LogisticRegression", {}).get("recall", 0.9132) * 100,
            models_data.get("LogisticRegression", {}).get("f1_score", 0.9032) * 100,
        ]
        nb_scores = [
            models_data.get("MultinomialNB", {}).get("accuracy", 0.8844) * 100,
            models_data.get("MultinomialNB", {}).get("precision", 0.8794) * 100,
            models_data.get("MultinomialNB", {}).get("recall", 0.8921) * 100,
            models_data.get("MultinomialNB", {}).get("f1_score", 0.8857) * 100,
        ]

        x = np.arange(len(metric_names))
        width = 0.35

        fig_bar, ax_bar = plt.subplots(figsize=(10, 4.2), facecolor="#0B0B0F")
        ax_bar.set_facecolor("#14141B")

        rects1 = ax_bar.bar(x - width/2, lr_scores, width, label="Logistic Regression (Champion)", color="#FFB020", edgecolor="#0B0B0F")
        rects2 = ax_bar.bar(x + width/2, nb_scores, width, label="Multinomial Naive Bayes", color="#2EE59D", edgecolor="#0B0B0F")

        ax_bar.set_ylabel("Score (%)", color="#9A97A3", fontsize=11, labelpad=10)
        ax_bar.set_title("Held-Out Test Set Performance Benchmark", color="#F4EFE6", fontsize=13, pad=15, weight="bold")
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(metric_names, color="#F4EFE6", fontsize=11)
        ax_bar.tick_params(colors="#9A97A3")
        ax_bar.set_ylim(80, 96)
        ax_bar.grid(axis="y", linestyle="--", alpha=0.25, color="#9A97A3")
        ax_bar.legend(facecolor="#1C1C26", edgecolor="#3D3D52", labelcolor="#F4EFE6")

        # Value annotations
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax_bar.annotate(f"{height:.2f}%",
                                xy=(rect.get_x() + rect.get_width() / 2, height),
                                xytext=(0, 4),
                                textcoords="offset points",
                                ha="center", va="bottom",
                                color="#F4EFE6", fontsize=9, fontweight="bold")

        autolabel(rects1)
        autolabel(rects2)

        for spine in ax_bar.spines.values():
            spine.set_edgecolor("#272736")

        plt.tight_layout()
        st.pyplot(fig_bar, use_container_width=True)
        plt.close(fig_bar)

        st.markdown('<div class="sprocket-border"></div>', unsafe_allow_html=True)

        # Side-by-side Confusion Matrices
        st.markdown(
            """
            <h3 style="font-family:'Bebas Neue', sans-serif; font-size:1.8rem; letter-spacing:0.05em; color:var(--text-main); margin-bottom: 15px;">
                CONFUSION MATRIX HEATMAPS (HELD-OUT TEST SET · 9,917 SAMPLES)
            </h3>
            """,
            unsafe_allow_html=True
        )

        cm_col1, cm_col2 = st.columns(2)
        amber_cmap = LinearSegmentedColormap.from_list("proj_amber", ["#14141B", "#3D2B0F", "#996300", "#FFB020", "#FFE4A0"])

        with cm_col1:
            cm_lr = np.array(models_data["LogisticRegression"]["confusion_matrix"])
            fig_lr, ax_lr = plt.subplots(figsize=(5.5, 4.2), facecolor="#0B0B0F")
            ax_lr.set_facecolor("#14141B")
            sns.heatmap(
                cm_lr, annot=True, fmt="d", cmap=amber_cmap, cbar=False, ax=ax_lr,
                annot_kws={"size": 13, "weight": "bold", "color": "#F4EFE6"},
                xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"],
                linewidths=1.5, linecolor="#1C1C26"
            )
            ax_lr.set_title("Logistic Regression (Champion Cut)", fontsize=12, color="#FFB020", pad=12, weight="bold")
            ax_lr.set_xlabel("Predicted Label", color="#9A97A3", fontsize=10)
            ax_lr.set_ylabel("True Label", color="#9A97A3", fontsize=10)
            ax_lr.tick_params(colors="#F4EFE6")
            for sp in ax_lr.spines.values():
                sp.set_edgecolor("#272736")
            plt.tight_layout()
            st.pyplot(fig_lr, use_container_width=True)
            plt.close(fig_lr)

        with cm_col2:
            cm_nb = np.array(models_data["MultinomialNB"]["confusion_matrix"])
            fig_nb, ax_nb = plt.subplots(figsize=(5.5, 4.2), facecolor="#0B0B0F")
            ax_nb.set_facecolor("#14141B")
            sns.heatmap(
                cm_nb, annot=True, fmt="d", cmap=amber_cmap, cbar=False, ax=ax_nb,
                annot_kws={"size": 13, "weight": "bold", "color": "#F4EFE6"},
                xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"],
                linewidths=1.5, linecolor="#1C1C26"
            )
            ax_nb.set_title("Multinomial Naive Bayes", fontsize=12, color="#2EE59D", pad=12, weight="bold")
            ax_nb.set_xlabel("Predicted Label", color="#9A97A3", fontsize=10)
            ax_nb.set_ylabel("True Label", color="#9A97A3", fontsize=10)
            ax_nb.tick_params(colors="#F4EFE6")
            for sp in ax_nb.spines.values():
                sp.set_edgecolor("#272736")
            plt.tight_layout()
            st.pyplot(fig_nb, use_container_width=True)
            plt.close(fig_nb)

        # Classification reports in expanders
        st.markdown('<div class="sprocket-border"></div>', unsafe_allow_html=True)
        rep_col1, rep_col2 = st.columns(2)

        with rep_col1:
            with st.expander("📋 Detailed Logistic Regression Classification Report"):
                lr_rep = models_data["LogisticRegression"]["classification_report"]
                df_lr_rep = pd.DataFrame(lr_rep).transpose()
                st.dataframe(df_lr_rep.style.format("{:.4f}"), use_container_width=True)

        with rep_col2:
            with st.expander("📋 Detailed Multinomial Naive Bayes Classification Report"):
                nb_rep = models_data["MultinomialNB"]["classification_report"]
                df_nb_rep = pd.DataFrame(nb_rep).transpose()
                st.dataframe(df_nb_rep.style.format("{:.4f}"), use_container_width=True)

    # =========================================================================
    # TAB 4: ABOUT (NLP THEORY & VIVA PREPARATION)
    # =========================================================================
    with tab_about:
        st.markdown(
            """
            <div style="margin-bottom: 25px;">
                <h2 style="font-family:'Bebas Neue', sans-serif; font-size:2.2rem; letter-spacing:0.05em; color:var(--text-main); margin-bottom: 4px;">
                    THE CINEMATOGRAPHY OF NATURAL LANGUAGE PROCESSING
                </h2>
                <p style="font-size:0.95rem; color:var(--text-muted); margin:0;">
                    Comprehensive theoretical foundations, architectural decisions, and interview-ready concepts.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        a_col1, a_col2 = st.columns(2)

        with a_col1:
            st.markdown(
                """
                <div class="info-card">
                    <div class="info-header">01 · WHAT IS SENTIMENT ANALYSIS?</div>
                    <div class="info-body">
                        Sentiment analysis is the computational classification of opinion, subjectivity, and emotional polarity within text.
                        In movie criticism, reviews contain layered prose—praising technical achievements while criticizing screenwriting,
                        or deploying sarcasm and plot twists. Our goal is to extract the net orientation: <strong>Positive</strong> or <strong>Negative</strong>.
                    </div>
                </div>

                <div class="info-card">
                    <div class="info-header">02 · WHY TF-IDF OVER BAG-OF-WORDS?</div>
                    <div class="info-body">
                        Standard Count Vectorization (Bag-of-Words) merely counts word frequency, unfairly favoring universally frequent but uninformative words like <em>movie</em> or <em>character</em>.<br><br>
                        <strong>TF-IDF</strong> (Term Frequency · Inverse Document Frequency) downweights words appearing across every document while amplifying distinctive sentiment descriptors like <em>masterpiece</em>, <em>abysmal</em>, or <em>riveting</em>:
                        <br><br>
                        <code>TF-IDF(t, d) = TF(t, d) × log(N / DF(t))</code>
                    </div>
                </div>

                <div class="info-card">
                    <div class="info-header">03 · PRESERVING NEGATIONS IN PREPROCESSING</div>
                    <div class="info-body">
                        A notorious pitfall in naive NLP pipelines is indiscriminately stripping all NLTK/sklearn stop words. Default stop word corpora include negation words like <code>not</code>, <code>no</code>, <code>never</code>, and <code>nothing</code>.<br><br>
                        Stripping these reverses sentiment polarity: <em>"The film was not good"</em> becomes <em>"film good"</em>! ReviewSense explicitly protects 20+ negation qualifiers, preserving crucial polarity reversals.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with a_col2:
            st.markdown(
                f"""
                <div class="info-card">
                    <div class="info-header">04 · LOGISTIC REGRESSION (THE CHAMPION)</div>
                    <div class="info-body">
                        Logistic Regression models the log-odds of a review being positive as a linear combination of its TF-IDF feature weights:<br><br>
                        <code>P(Y=1|X) = 1 / (1 + e^-(w·X + b))</code><br><br>
                        Because TF-IDF features are dense with subtle n-gram correlations (e.g. <em>not bad</em>, <em>rarely seen</em>), Logistic Regression reached an exceptional <strong>{best_f1:.2f}% F1-score</strong> on the held-out test set.
                    </div>
                </div>

                <div class="info-card">
                    <div class="info-header">05 · MULTINOMIAL NAIVE BAYES (THE SPEED DEMON)</div>
                    <div class="info-body">
                        Multinomial Naive Bayes applies Bayes' Theorem under the "naive" conditional independence assumption—assuming every word occurs independently of all others given the class:<br><br>
                        <code>P(C|w1, ..., wn) ∝ P(C) ∏ P(wi|C)</code><br><br>
                        While it trained in just 0.05 seconds, its independence assumption slightly penalizes phrases where word pairing alters meaning, finishing closely behind at <strong>{models_data.get('MultinomialNB', {}).get('f1_score', 0.8857)*100:.2f}% F1</strong>.
                    </div>
                </div>

                <div class="info-card">
                    <div class="info-header">06 · PREVENTING DATA LEAKAGE</div>
                    <div class="info-body">
                        A common error in student projects is fitting the vectorizer on the whole dataset before splitting into train and test sets. This contaminates the model with vocabulary distribution knowledge from the test set.<br><br>
                        ReviewSense splits raw texts first, fits <code>TfidfVectorizer</code> exclusively on the 80% train partition, and only transforms the 20% test slice.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )




if __name__ == "__main__":
    main()
