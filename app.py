import streamlit as st
import pandas as pd
import numpy as np
import requests
import re
from sklearn.cluster import KMeans
from urllib.parse import quote

st.title("Smart Job Router (UK Reliable Version)")

uploaded_file = st.file_uploader("Upload Jobs Excel")
engineers = st.number_input("Number of engineers", min_value=1, value=4)


# ---------------- POSTCODE EXTRACTION (LAST COMMA) ----------------
def extract_postcode(text):
    if pd.isna(text):
        return None

    parts = [p.strip() for p in str(text).split(",") if p.strip()]

    if not parts:
        return None

    return parts[-1].upper()


# ---------------- UK GEOCODING (POSTCODES.IO) ----------------
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


# ---------------- MAP LINK ----------------
def maps_link(addresses):
    clean = [str(a) for a in addresses if pd.notna(a)]
    return "https://www.google.com/maps/dir/" + "/".join([quote(a) for a in clean])


# ---------------- MAIN APP ----------------
if uploaded_file:

    df = pd.read_excel(uploaded_file)
    df.columns = df.columns.str.strip()

    # detect address column
    address_col = next((c for c in df.columns if "address" in c.lower()), None)

    if not address_col:
        st.error(f"No address column found. Columns: {list(df.columns)}")
        st.stop()

    st.write("Using column:", address_col)

    # ---------------- EXTRACT POSTCODES ----------------
    df["postcode"] = df[address_col].apply(extract_postcode)

    st.write("Sample extracted postcodes:")
    st.write(df["postcode"].head(10))

    # ---------------- GEOCODING ----------------
    lat_list = []
    lon_list = []
    failed = []

    for i, row in df.iterrows():

        postcode = row["postcode"]

        lat, lon = geocode_postcode(postcode)

        if lat is None:
            failed.append(i)

        lat_list.append(lat)
        lon_list.append(lon)

    df["lat"] = lat_list
    df["lon"] = lon_list

    # ---------------- REPORT ----------------
    st.warning(f"Failed to geocode: {len(failed)} jobs")

    df = df.dropna(subset=["lat", "lon"]).reset_index(drop=True)

    st.write(f"Jobs successfully routed: {len(df)}")

    if len(df) == 0:
        st.error("No valid jobs after geocoding.")
        st.stop()

    # ---------------- CLUSTERING ----------------
    if len(df) < engineers:
        st.error("Not enough jobs for number of engineers.")
        st.stop()

    coords = np.radians(df[['lat', 'lon']])
    kmeans = KMeans(n_clusters=engineers, random_state=0, n_init=10)

    df["Engineer"] = kmeans.fit_predict(coords)

    # ---------------- BALANCING ----------------
    max_jobs = int(np.ceil(len(df) / engineers))

    for i in range(engineers):
        cluster = df[df["Engineer"] == i]

        if len(cluster) > max_jobs:
            extra = cluster.iloc[max_jobs:]

            for idx, row in extra.iterrows():
                distances = (
                    (df['lat'] - row['lat'])**2 +
                    (df['lon'] - row['lon'])**2
                )

                nearest = distances.idxmin()
                df.at[idx, "Engineer"] = df.loc[nearest, "Engineer"]

    st.success("Routing complete!")

    # ---------------- OUTPUT ----------------
    for i in range(engineers):

        st.subheader(f"Engineer {i+1}")

        eng_jobs = df[df["Engineer"] == i]

        st.dataframe(
            eng_jobs[[address_col, "postcode", "lat", "lon"]]
        )

        if not eng_jobs.empty:
            link = maps_link(eng_jobs[address_col].tolist())
            st.markdown(f"[Open Route in Google Maps]({link})")
