
import streamlit as st
import pandas as pd
import requests
import numpy as np
from sklearn.cluster import KMeans
from urllib.parse import quote

st.title("Smart Job Router (Pro)")

uploaded_file = st.file_uploader("Upload Jobs Excel")

engineers = st.number_input("Number of engineers", min_value=1, value=4)


# --- Geocode ---
@st.cache_data
def geocode(address):
    if pd.isna(address):
        return None, None

    url = f"https://nominatim.openstreetmap.org/search?format=json&q={address}"
    headers = {"User-Agent": "smart-job-router"}

    try:
        r = requests.get(url, headers=headers, timeout=10).json()
        if not r:
            return None, None
        return float(r[0]['lat']), float(r[0]['lon'])
    except:
        return None, None


def maps_link(addresses):
    return "https://www.google.com/maps/dir/" + "/".join([quote(a) for a in addresses])


# ---------------- MAIN APP ----------------
if uploaded_file:

    df = pd.read_excel(uploaded_file)

    possible_cols = ['Address', 'address', 'Full Address', 'Job Address']
    address_col = next((col for col in possible_cols if col in df.columns), None)

    if not address_col:
        st.error("No address column found")

    elif 'Slot' not in df.columns:
        st.error("Missing 'Slot' column (AM/PM)")

    else:
        st.write("Geocoding addresses...")

        df[['lat','lon']] = df[address_col].apply(
            lambda x: pd.Series(geocode(x))
        )

        df = df.dropna(subset=['lat', 'lon'])

        kmeans = KMeans(n_clusters=engineers, random_state=0)
        df['Engineer'] = kmeans.fit_predict(df[['lat','lon']])

        max_jobs = int(np.ceil(len(df) / engineers))

        for i in range(engineers):
            over = df[df['Engineer'] == i]

            if len(over) > max_jobs:
                extra = over.iloc[max_jobs:]

                for idx, row in extra.iterrows():
                    distances = ((df['lat'] - row['lat'])**2 +
                                 (df['lon'] - row['lon'])**2)

                    nearest_cluster = distances.idxmin()
                    df.at[idx, 'Engineer'] = df.loc[nearest_cluster, 'Engineer']

        st.success("Routing complete!")
