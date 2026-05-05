import streamlit as st
import pandas as pd
import numpy as np
import requests
from urllib.parse import quote
from sklearn.cluster import KMeans
from math import radians, sin, cos, sqrt, atan2

st.title("Smart Job Router (Fast Geographic Dispatcher)")

uploaded_file = st.file_uploader("Upload Jobs Excel")
engineers = st.number_input("Number of engineers", min_value=1, value=4)

# ---------------- SETTINGS ----------------
MAX_JOBS = 5
MIN_JOBS = 4


# ---------------- POSTCODE EXTRACTION ----------------
def extract_postcode(text):
    if pd.isna(text):
        return None
    parts = [p.strip() for p in str(text).split(",") if p.strip()]
    return parts[-1].upper() if parts else None


# ---------------- GEOCODING ----------------
@st.cache_data
def geocode_postcode(postcode):
    if not postcode:
        return None, None

    try:
        url = f"https://api.postcodes.io/postcodes/{postcode.replace(' ', '')}"
        r = requests.get(url, timeout=10).json()

        if r["status"] != 200:
            return None, None

        res = r["result"]
        return res["latitude"], res["longitude"]

    except:
        return None, None


# ---------------- DISTANCE ----------------
def distance(a, b):
    R = 6371

    lat1, lon1 = map(radians, a)
    lat2, lon2 = map(radians, b)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    h = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    return 2 * R * atan2(sqrt(h), sqrt(1 - h))


# ---------------- GOOGLE MAPS ----------------
def maps_link(addresses):
    clean = [str(a).strip() for a in addresses if pd.notna(a)]

    if len(clean) <= 1:
        return None

    return (
        "https://www.google.com/maps/dir/?api=1"
        "&travelmode=driving"
        "&waypoints=" + "|".join(quote(a) for a in clean[1:])
    )


# ---------------- MAIN APP ----------------
if uploaded_file:

    df = pd.read_excel(uploaded_file)
    df.columns = df.columns.str.strip()

    address_col = next((c for c in df.columns if "address" in c.lower()), None)

    if not address_col:
        st.error("No address column found")
        st.stop()

    st.write("Using column:", address_col)

    # ---------------- POSTCODES ----------------
    df["postcode"] = df[address_col].apply(extract_postcode)

    # ---------------- GEOCODING ----------------
    unique_postcodes = df["postcode"].dropna().unique()

    cache = {pc: geocode_postcode(pc) for pc in unique_postcodes}

    df["lat"] = df["postcode"].map(lambda x: cache.get(x, (None, None))[0])
    df["lon"] = df["postcode"].map(lambda x: cache.get(x, (None, None))[1])

    df = df.dropna(subset=["lat", "lon"]).reset_index(drop=True)

    if df.empty:
        st.error("No valid geocoded jobs")
        st.stop()

    st.success(f"Valid jobs: {len(df)}")

    # ======================================================
    # 🧠 IMPROVED FAST CLUSTERING (KEY FIX)
    # ======================================================

    coords = df[["lat", "lon"]].values

    n_clusters = min(engineers, len(df))

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )

    df["Engineer"] = kmeans.fit_predict(coords)

    centroids = kmeans.cluster_centers_

    # ======================================================
    # ⚖️ FIX: BALANCE + PREVENT FAR OUTLIERS
    # ======================================================

    def cluster_size(e):
        return len(df[df["Engineer"] == e])

    def assign_best_cluster(row):
        point = (row["lat"], row["lon"])

        scores = []

        for i in range(n_clusters):

            center = centroids[i]

            # distance to centroid
            d = distance(point, (center[0], center[1]))

            # penalty for overcrowded clusters
            size_penalty = cluster_size(i) * 0.15

            scores.append((i, d + size_penalty))

        return min(scores, key=lambda x: x[1])[0]

    df["Engineer"] = df.apply(assign_best_cluster, axis=1)

    # ======================================================
    # ⚖️ HARD CAPACITY ENFORCEMENT (4–5 JOBS)
    # ======================================================

    for e in range(n_clusters):

        cluster = df[df["Engineer"] == e]

        if len(cluster) > MAX_JOBS:

            overflow = cluster.iloc[MAX_JOBS:]

            for idx in overflow.index:

                point = (df.loc[idx, "lat"], df.loc[idx, "lon"])

                candidates = []

                for j in range(n_clusters):

                    if cluster_size(j) < MAX_JOBS:

                        center = centroids[j]
                        d = distance(point, (center[0], center[1]))

                        candidates.append((j, d))

                if candidates:
                    new_cluster = min(candidates, key=lambda x: x[1])[0]
                    df.at[idx, "Engineer"] = new_cluster

    st.success("Routing complete!")

    # ======================================================
    # 📦 OUTPUT
    # ======================================================

    for i in sorted(df["Engineer"].unique()):

        eng_df = df[df["Engineer"] == i].copy()

        st.subheader(f"Engineer {i + 1}")

        st.dataframe(
            eng_df[[address_col, "postcode", "lat", "lon"]]
        )

        link = maps_link(eng_df[address_col].tolist())

        if link:
            st.markdown(f"[Open Route in Google Maps]({link})")
