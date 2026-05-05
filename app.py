import streamlit as st
import pandas as pd
import numpy as np
import requests
from urllib.parse import quote
from sklearn.cluster import KMeans
from math import radians, sin, cos, sqrt, atan2

st.title("Smart Job Router (Capacity-Aware Geographic Dispatcher)")

uploaded_file = st.file_uploader("Upload Jobs Excel")
engineers = st.number_input("Number of engineers", min_value=1, value=4)

# ---------------- SETTINGS ----------------
MIN_JOBS = 4
MAX_JOBS = 5


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


# ---------------- DISTANCE (HAVERSINE) ----------------
def distance(a, b):
    R = 6371

    lat1, lon1 = map(radians, a)
    lat2, lon2 = map(radians, b)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    h = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    return 2 * R * atan2(sqrt(h), sqrt(1 - h))


# ---------------- CLUSTER SPREAD (NEW FIX) ----------------
def cluster_spread(cluster):
    """Measures how geographically tight a cluster is"""
    if len(cluster) <= 1:
        return 0

    pts = cluster[["lat", "lon"]].values
    center = np.mean(pts, axis=0)

    return np.mean([
        distance((p[0], p[1]), (center[0], center[1]))
        for p in pts
    ])


# ---------------- ROUTE ORDERING (LOCAL ONLY) ----------------
def order_route(df):
    if len(df) <= 2:
        return df

    df = df.copy().reset_index(drop=True)
    remaining = df.copy()
    route = []

    current = remaining.iloc[0]
    route.append(current)
    remaining = remaining.drop(0)

    while len(remaining) > 0:

        current_point = (current["lat"], current["lon"])

        remaining["dist"] = remaining.apply(
            lambda row: distance(current_point, (row["lat"], row["lon"])),
            axis=1
        )

        next_idx = remaining["dist"].idxmin()
        current = remaining.loc[next_idx]

        route.append(current)
        remaining = remaining.drop(next_idx)

    return pd.DataFrame(route).drop(columns=["dist"], errors="ignore")


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
        st.error("No address column found.")
        st.stop()

    st.write("Using column:", address_col)

    # ---------------- POSTCODES ----------------
    df["postcode"] = df[address_col].apply(extract_postcode)

    # ---------------- BATCH GEOCODING ----------------
    unique_postcodes = df["postcode"].dropna().unique()

    cache = {}
    for pc in unique_postcodes:
        cache[pc] = geocode_postcode(pc)

    df["lat"] = df["postcode"].map(lambda x: cache.get(x, (None, None))[0])
    df["lon"] = df["postcode"].map(lambda x: cache.get(x, (None, None))[1])

    df = df.dropna(subset=["lat", "lon"]).reset_index(drop=True)

    if df.empty:
        st.error("No valid geocoded jobs.")
        st.stop()

    st.success(f"Valid jobs: {len(df)}")

    # ======================================================
    # 🧠 GEOGRAPHIC CLUSTERING
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

    # helper
    def cluster_size(e):
        return len(df[df["Engineer"] == e])

    # ======================================================
    # ⚖️ IMPROVED CAPACITY BALANCING (FIXED ISSUE)
    # ======================================================

    for e in range(n_clusters):

        cluster = df[df["Engineer"] == e]

        if len(cluster) > MAX_JOBS:

            overflow = cluster.iloc[MAX_JOBS:]

            for idx, row in overflow.iterrows():

                point = (row["lat"], row["lon"])

                candidates = []

                for j in range(n_clusters):

                    if cluster_size(j) < MAX_JOBS:

                        center = centroids[j]

                        cluster_j = df[df["Engineer"] == j]

                        # NEW: combine centroid distance + cluster compactness
                        centroid_dist = distance(point, (center[0], center[1]))
                        spread_penalty = cluster_spread(cluster_j)

                        score = centroid_dist + (0.5 * spread_penalty)

                        candidates.append((j, score))

                if candidates:
                    new_cluster = min(candidates, key=lambda x: x[1])[0]
                    df.at[idx, "Engineer"] = new_cluster

    # ======================================================
    # ⚖️ FILL UNDERFILLED CLUSTERS
    # ======================================================

    for e in range(n_clusters):

        while cluster_size(e) < MIN_JOBS:

            donors = [
                j for j in range(n_clusters)
                if cluster_size(j) > MAX_JOBS
            ]

            if not donors:
                break

            donor = donors[0]
            donor_df = df[df["Engineer"] == donor]

            move_idx = donor_df.index[0]
            df.at[move_idx, "Engineer"] = e

    st.success("Assignment complete!")

    # ======================================================
    # 📦 OUTPUT
    # ======================================================

    for e in sorted(df["Engineer"].unique()):

        eng_df = df[df["Engineer"] == e].copy()

        eng_df = order_route(eng_df)

        st.subheader(f"Engineer {e + 1}")

        st.dataframe(
            eng_df[[address_col, "postcode", "lat", "lon"]]
        )

        link = maps_link(eng_df[address_col].tolist())
        if link:
            st.markdown(f"[Open Route in Google Maps]({link})")
