import streamlit as st
import pandas as pd
import requests
import numpy as np
from sklearn.cluster import KMeans
from urllib.parse import quote
import time

st.title("Smart Job Router (Pro - Postcode Optimised)")

uploaded_file = st.file_uploader("Upload Jobs Excel")
engineers = st.number_input("Number of engineers", min_value=1, value=4)


# ---------------- GEOLOCATION (POSTCODE FIRST) ----------------
@st.cache_data
def geocode_postcode(postcode):
    if pd.isna(postcode):
        return None, None

    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "format": "json",
        "q": postcode,
        "countrycodes": "gb"   # UK optimisation
    }

    headers = {"User-Agent": "smart-job-router"}

    try:
        r = requests.get(url, params=params, headers=headers, timeout=10).json()
        if not r:
            return None, None
        return float(r[0]["lat"]), float(r[0]["lon"])
    except:
        return None, None


# fallback full address
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


# ---------------- ROUTE LINK ----------------
def maps_link(addresses):
    clean = [str(a) for a in addresses if pd.notna(a)]
    return "https://www.google.com/maps/dir/" + "/".join([quote(a) for a in clean])


# ---------------- MAIN ----------------
if uploaded_file:

    df = pd.read_excel(uploaded_file)
    df.columns = df.columns.str.strip()

    # ---------------- COLUMN DETECTION ----------------
    postcode_col = next((c for c in df.columns if "post" in c.lower()), None)
    address_col = next((c for c in df.columns if "address" in c.lower()), None)

    if not postcode_col:
        st.error("No postcode column found (must contain 'postcode' in column name)")
        st.stop()

    st.write("Using postcode:", postcode_col)

    # ---------------- GEOCODING ----------------
    lat_list = []
    lon_list = []
    failed = []

    for i, row in df.iterrows():

        postcode = row[postcode_col]
        address = row[address_col] if address_col else None

        lat, lon = geocode_postcode(postcode)

        # fallback if postcode fails
        if lat is None:
            lat, lon = geocode_address(address)

        if lat is None:
            failed.append(i)

        lat_list.append(lat)
        lon_list.append(lon)

        time.sleep(1)  # avoids API blocking

    df["lat"] = lat_list
    df["lon"] = lon_list

    # ---------------- REPORT FAILURES ----------------
    if len(failed) > 0:
        st.warning(f"{len(failed)} jobs could not be geocoded")

    df = df.dropna(subset=["lat", "lon"]).reset_index(drop=True)

    st.write(f"Jobs being routed: {len(df)}")

    # ---------------- CLUSTERING ----------------
    if len(df) < engineers:
        st.error("Not enough valid jobs for number of engineers")
        st.stop()

    coords = np.radians(df[['lat', 'lon']])
    kmeans = KMeans(n_clusters=engineers, random_state=0, n_init=10)
    df["Engineer"] = kmeans.fit_predict(coords)

    # ---------------- BALANCE ----------------
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

        st.dataframe(eng_jobs[[postcode_col] + ([address_col] if address_col else [])])

        if not eng_jobs.empty:
            link = maps_link(eng_jobs[address_col].tolist() if address_col else eng_jobs[postcode_col].tolist())
            st.markdown(f"[Open Route in Google Maps]({link})")
