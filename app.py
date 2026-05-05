import streamlit as st
import pandas as pd
import requests
import numpy as np
import re
from sklearn.cluster import KMeans
from urllib.parse import quote
import time

st.title("Smart Job Router (Pro - UK Postcode Aware)")

uploaded_file = st.file_uploader("Upload Jobs Excel")
engineers = st.number_input("Number of engineers", min_value=1, value=4)


# ---------------- UK POSTCODE EXTRACT ----------------
def extract_postcode(text):
    if pd.isna(text):
        return None

    text = str(text).upper()

    # UK postcode pattern
    match = re.search(r'[A-Z]{1,2}\d[A-Z\d]?\s?\d[A-Z]{2}', text)
    return match.group(0) if match else None


# ---------------- GEOCODING ----------------
@st.cache_data
def geocode_postcode(postcode):
    if not postcode:
        return None, None

    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "format": "json",
        "q": postcode,
        "countrycodes": "gb"
    }

    headers = {"User-Agent": "smart-job-router"}

    try:
        r = requests.get(url, params=params, headers=headers, timeout=10).json()
        if not r:
            return None, None
        return float(r[0]["lat"]), float(r[0]["lon"])
    except:
        return None, None


@st.cache_data
def geocode_address(address):
    if pd.isna(address):
        return None, None

    url = "https://nominatim.openstreetmap.org/search"
    params = {"format": "json", "q": address}

    headers = {"User-Agent": "smart-job-router"}

    try:
        r = requests.get(url, params=params, headers=headers, timeout=10).json()
        if not r:
            return None, None
        return float(r[0]["lat"]), float(r[0]["lon"])
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

    st.write("Extracted postcodes:", df["postcode"].notna().sum())

    lat_list = []
    lon_list = []
    failed = []

    # ---------------- GEOCODING LOOP ----------------
    for i, row in df.iterrows():

        postcode = row["postcode"]
        address = row[address_col]

        # 1. try postcode
        lat, lon = geocode_postcode(postcode)

        # 2. fallback to full address
        if lat is None:
            lat, lon = geocode_address(address)

        if lat is None:
            failed.append(i)

        lat_list.append(lat)
        lon_list.append(lon)

        time.sleep(1)  # avoid API blocking

    df["lat"] = lat_list
    df["lon"] = lon_list

    # ---------------- REPORT ISSUES ----------------
    st.warning(f"Failed to geocode: {len(failed)} jobs")

    df = df.dropna(subset=["lat", "lon"]).reset_index(drop=True)

    st.write(f"Jobs successfully routed: {len(df)}")

    if len(df) == 0:
        st.error("No valid jobs to route after geocoding.")
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
