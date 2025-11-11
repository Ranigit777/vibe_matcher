# ==========================================================
# 🎧 Streamlit Vibe Matcher App
# ----------------------------------------------------------
# Type a vibe (e.g., "energetic urban chic") and get top-3
# fashion item recommendations based on description embeddings.
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time

# Optional: use OpenAI embeddings if you have an API key
USE_REAL_EMBEDDINGS = False

if USE_REAL_EMBEDDINGS:
    from openai import OpenAI
    import os
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ----------------------------------------------------------
# Step 1: Create sample dataset
# ----------------------------------------------------------
data = [
    {"name": "Boho Dress", "desc": "Flowy, earthy tones for festival vibes", "vibes": ["boho", "casual"]},
    {"name": "Urban Jacket", "desc": "Sleek black leather, perfect for city nights", "vibes": ["urban", "chic"]},
    {"name": "Cozy Sweater", "desc": "Warm, soft knit ideal for coffee shop mornings", "vibes": ["cozy", "minimal"]},
    {"name": "Sporty Sneakers", "desc": "Energetic design built for comfort and movement", "vibes": ["sporty", "energetic"]},
    {"name": "Vintage Denim", "desc": "Classic blue jeans with a timeless retro vibe", "vibes": ["vintage", "casual"]},
    {"name": "Elegant Gown", "desc": "Sophisticated evening wear with luxurious fabric", "vibes": ["elegant", "formal"]},
    {"name": "Street Hoodie", "desc": "Casual hoodie with graffiti-inspired design", "vibes": ["street", "urban"]},
    {"name": "Beach Shorts", "desc": "Light and airy shorts for summer days", "vibes": ["beach", "relaxed"]},
]

df = pd.DataFrame(data)

# ----------------------------------------------------------
# Step 2: Get embeddings (random or real)
# ----------------------------------------------------------
def get_embedding(text):
    """Generate embedding from OpenAI or random for demo."""
    if USE_REAL_EMBEDDINGS:
        response = client.embeddings.create(model="text-embedding-ada-002", input=text)
        return response.data[0].embedding
    else:
        np.random.seed(abs(hash(text)) % (10**6))
        return np.random.rand(1536)

df["embedding"] = df["desc"].apply(get_embedding)

# ----------------------------------------------------------
# Step 3: Streamlit UI
# ----------------------------------------------------------
st.set_page_config(page_title="🎧 Vibe Matcher", layout="centered")
st.title("🎧 Vibe Matcher – Fashion Recommender")

st.write("Find products that match your **vibe**! Try vibes like:")
st.code("energetic urban chic, cozy winter look, boho festival style")

query = st.text_input("Enter your vibe:", "")

if st.button("Find Matches"):
    if query.strip() == "":
        st.warning("Please enter a vibe before searching!")
    else:
        start = time.time()
        q_embed = get_embedding(query)
        sims = cosine_similarity([q_embed], df["embedding"].tolist())[0]
        df["similarity"] = sims
        top3 = df.sort_values("similarity", ascending=False).head(3)
        latency = round(time.time() - start, 3)

        st.success(f"Results for **'{query}'** (took {latency}s):")
        for _, row in top3.iterrows():
            st.markdown(f"**{row['name']}** — `{row['similarity']:.3f}`")
            st.write(row["desc"])
            st.write("---")

        if top3["similarity"].max() < 0.5:
            st.info("No strong match found — maybe try a different vibe?")

st.caption("Prototype built with ❤️ using Streamlit and Python.")
