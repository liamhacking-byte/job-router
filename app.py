import streamlit as st
import pandas as pd
import numpy as np
import requests
from urllib.parse import quote

st.title("Smart Job Router (Geographic Dispatch Engine)")

uploaded_file = st.file_uploader("Upload Jobs Excel")
engineers = st.number_input("Number of engineers", min_value=1, value=4)


# ---------------- POSTCODE EXTRACTION ----------------
def extract_postcode(text):
    if pd.isna(text):
        return None

    parts = [p.strip() for p in str(text).split(",") if p.strip()]
    if not parts:
        return None

    return parts[-1].upper()


# ---------------- UK POSTCODE GEOCODING ----------------
@st.cache_data
def geocode_postcode(postcode):
    if not postcode:
        return None, None

    try:
        url = f"https://api.postcodes.io/postcodes/{postcode.replace(' ', '')}"
        r = requests.get(url, timeout=10).json()

        if r["status"] != 200:
            return None, None

        result = r["result"]
        return result["latitude"], result["longitude"]

    except:
        return None, None


# ---------------- ROUTING HELPERS ----------------
def distance(a, b):
    return (a[0] - b[0])**2 + (a[1] - b[1])**2


def order_route(df):
    """Nearest-neighbour route ordering"""
    if len(df) <= 1:
        return df

    remaining = df.copy()
    route = []

    current = remaining.iloc[0]
    route.append(current)
    remaining = remaining.drop(current.name)

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


def maps_link(addresses):
    clean = [str(a) for a in addresses if pd.notna(a)]
    return "https://www.google.com/maps/dir/" + "/".join([quote(a) for a in clean])


# ---------------- MAIN APP ----------------
if uploaded_file:

    df = pd.read_excel(uploaded_file)
    df.columns = df.columns.str.strip()

    address_col = next((c for c in df.columns if "address" in c.lower()), None)

    if not address_col:
        st.error(f"No address column found. Columns: {list(df.columns)}")
        st.stop()

    st.write("Using column:", address_col)

    # ---------------- EXTRACT POSTCODES ----------------
    df["postcode"] = df[address_col].apply(extract_postcode)

    st.write("Sample postcodes:")
    st.write(df["postcode"].head(10))

    # ---------------- GEOCODING ----------------
    lat_list = []
    lon_list = []
    failed = []

    for _, row in df.iterrows():

        lat, lon = geocode_postcode(row["postcode"])

        if lat is None:
            failed.append(True)

        lat_list.append(lat)
        lon_list.append(lon)

    df["lat"] = lat_list
    df["lon"] = lon_list

    st.warning(f"Failed geocoding: {len(failed)} jobs")

    df = df.dropna(subset=["lat", "lon"]).reset_index(drop=True)

    if len(df) == 0:
        st.error("No valid jobs after geocoding.")
        st.stop()

    st.write(f"Valid jobs: {len(df)}")

    # =========================================================
    # 🧠 GEOGRAPHIC SEED-BASED CLUSTERING (FIX FOR YOUR ISSUE)
    # =========================================================

    df = df.reset_index(drop=True)

    # 1. pick evenly spaced geographic seeds
    seed_indices = np.linspace(0, len(df) - 1, engineers, dtype=int)
    seeds = df.iloc[seed_indices]

    centers = list(zip(seeds["lat"], seeds["lon"]))

    df["Engineer"] = -1

    # 2. assign each job to nearest seed
    for idx, row in df.iterrows():

        point = (row["lat"], row["lon"])

        distances = [
            (i, distance(point, c))
            for i, c in enumerate(centers)
        ]

        df.at[idx, "Engineer"] = min(distances, key=lambda x: x[1])[0]

    # 3. balance workloads gently (without breaking geography)
    max_jobs = int(np.ceil(len(df) / engineers))

    for i in range(engineers):

        cluster = df[df["Engineer"] == i]

        if len(cluster) > max_jobs:

            excess = cluster.iloc[max_jobs:]

            for idx, row in excess.iterrows():

                point = (row["lat"], row["lon"])

                candidates = []

                for j in range(engineers):

                    if len(df[df["Engineer"] == j]) < max_jobs:
                        center = centers[j]
                        dist = distance(point, center)
                        candidates.append((j, dist))

                if candidates:
                    new_eng = min(candidates, key=lambda x: x[1])[0]
                    df.at[idx, "Engineer"] = new_eng

    st.success("Routing complete!")

    # ---------------- OUTPUT ----------------
    for i in range(engineers):

        eng_df = df[df["Engineer"] == i].copy()
        eng_df = order_route(eng_df)

        st.subheader(f"Engineer {i+1}")

        st.dataframe(
            eng_df[[address_col, "postcode", "lat", "lon"]]
        )

        if not eng_df.empty:
            link = maps_link(eng_df[address_col].tolist())
            st.markdown(f"[Open Route in Google Maps]({link})")
