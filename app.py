import streamlit as st
import pandas as pd
import requests
import numpy as np
from sklearn.cluster import KMeans
from urllib.parse import quote

st.title("Smart Job Router (Pro)")

uploaded_file = st.file_uploader("Upload Jobs Excel")
engineers = st.number_input("Number of engineers", min_value=1, value=4)


# ---------------- GEOLOCATION ----------------
@st.cache_data
def geocode(address):
    if pd.isna(address):
        return None, None

    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "format": "json",
        "q": address
    }

    headers = {"User-Agent": "smart-job-router"}

    try:
        r = requests.get(url, params=params, headers=headers, timeout=10).json()
        if not r:
            return None, None
        return float(r[0]["lat"]), float(r[0]["lon"])
    except:
        return None, None


# ---------------- MAPS ROUTE ----------------
def maps_link(addresses):
    clean = [str(a) for a in addresses if pd.notna(a)]
    return "https://www.google.com/maps/dir/" + "/".join([quote(a) for a in clean])


# ---------------- MAIN APP ----------------
if uploaded_file:

    df = pd.read_excel(uploaded_file)
    df.columns = df.columns.str.strip()

    # detect address column
    possible_cols = ['Address', 'address', 'Full Address', 'Job Address']
    address_col = next((col for col in possible_cols if col in df.columns), None)

    if not address_col:
        st.error("No address column found in Excel file.")

    elif 'Slot' not in df.columns:
        st.error("Missing 'Slot' column (must be AM/PM).")

    else:
        st.write("Geocoding addresses... (this may take a moment)")

        # geocode
        df[['lat', 'lon']] = df[address_col].apply(
            lambda x: pd.Series(geocode(x))
        )

        df = df.dropna(subset=['lat', 'lon']).reset_index(drop=True)

        if len(df) < engineers:
            st.error("Not enough jobs for number of engineers.")
            st.stop()

        # ---------------- CLUSTERING ----------------
        coords = np.radians(df[['lat', 'lon']])
        kmeans = KMeans(n_clusters=engineers, random_state=0, n_init=10)

        df['Engineer'] = kmeans.fit_predict(coords)

        # ---------------- BALANCING ----------------
        max_jobs = int(np.ceil(len(df) / engineers))

        for i in range(engineers):
            cluster = df[df['Engineer'] == i]

            if len(cluster) > max_jobs:
                excess = cluster.iloc[max_jobs:]

                for idx, row in excess.iterrows():
                    distances = (
                        (df['lat'] - row['lat'])**2 +
                        (df['lon'] - row['lon'])**2
                    )

                    nearest = distances.idxmin()
                    df.at[idx, 'Engineer'] = df.loc[nearest, 'Engineer']

        st.success("Routing complete!")

        # ---------------- OUTPUT ----------------
        for i in range(engineers):

            st.subheader(f"Engineer {i+1}")

            eng_jobs = df[df['Engineer'] == i]

            am = eng_jobs[eng_jobs['Slot'] == 'AM']
            pm = eng_jobs[eng_jobs['Slot'] == 'PM']

            route = pd.concat([am, pm])

            st.dataframe(route[[address_col, 'Slot']])

            if not route.empty:
                link = maps_link(route[address_col].tolist())
                st.markdown(f"[Open Route in Google Maps]({link})")
