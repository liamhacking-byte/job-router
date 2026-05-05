import streamlit as st
import pandas as pd
import numpy as np
import requests
from urllib.parse import quote

st.title("Smart Job Router (Dispatch Optimised)")

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


# ---------------- UK GEOCODING ----------------
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


# ---------------- DISTANCE (simple squared euclidean) ----------------
def distance(a, b):
    return (a[0] - b[0])**2 + (a[1] - b[1])**2


# ---------------- GREEDY ROUTE ORDER ----------------
def order_route(df):
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


# ---------------- MAP LINK ----------------
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

    st.write("Using:", address_col)

    # ---------------- POSTCODES ----------------
    df["postcode"] = df[address_col].apply(extract_postcode)

    st.write("Extracted postcodes sample:")
    st.write(df["postcode"].head(10))

    # ---------------- GEOCODING ----------------
    lat_list = []
    lon_list = []
    failed = []

    for i, row in df.iterrows():

        lat, lon = geocode_postcode(row["postcode"])

        if lat is None:
            failed.append(i)

        lat_list.append(lat)
        lon_list.append(lon)

    df["lat"] = lat_list
    df["lon"] = lon_list

    st.warning(f"Failed geocoding: {len(failed)} jobs")

    df = df.dropna(subset=["lat", "lon"]).reset_index(drop=True)

    st.write(f"Valid jobs: {len(df)}")

    if len(df) == 0:
        st.error("No valid jobs found.")
        st.stop()

    # ---------------- FAIR DISTRIBUTION (CORE LOGIC) ----------------
    df = df.sort_values(["lat"]).reset_index(drop=True)
    df["Engineer"] = np.arange(len(df)) % engineers

    # ---------------- BUILD ROUTES ----------------
    for i in range(engineers):

        st.subheader(f"Engineer {i+1}")

        eng_df = df[df["Engineer"] == i].copy()

        # order route geographically (nearest neighbour)
        eng_df = order_route(eng_df)

        st.dataframe(eng_df[[address_col, "postcode", "lat", "lon"]])

        if not eng_df.empty:
            link = maps_link(eng_df[address_col].tolist())
            st.markdown(f"[Open Route in Google Maps]({link})")
