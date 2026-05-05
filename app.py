import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.cluster import KMeans
from urllib.parse import quote

st.title("Smart Job Router (Geo-Optimised Dispatch)")

uploaded_file = st.file_uploader("Upload Jobs Excel")
engineers = st.number_input("Number of engineers", min_value=1, value=4)


# ---------------- POSTCODE EXTRACTION ----------------
def extract_postcode(text):
    if pd.isna(text):
        return None
    parts = [p.strip() for p in str(text).split(",") if p.strip()]
    return parts[-1].upper() if parts else None


# ---------------- GEOCODING ----------------
def geocode_postcode(pc):
    if not pc:
        return None, None

    try:
        url = f"https://api.postcodes.io/postcodes/{pc.replace(' ', '')}"
        r = requests.get(url, timeout=10).json()

        if r["status"] != 200:
            return None, None

        res = r["result"]
        return res["latitude"], res["longitude"]

    except:
        return None, None


# ---------------- ROUTE ORDER (NEAREST NEIGHBOUR) ----------------
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
            lambda r: (r["lat"] - current_point[0])**2 + (r["lon"] - current_point[1])**2,
            axis=1
        )

        nxt = remaining["dist"].idxmin()
        current = remaining.loc[nxt]

        route.append(current)
        remaining = remaining.drop(nxt)

    return pd.DataFrame(route).drop(columns=["dist"], errors="ignore")


# ---------------- MAP LINK ----------------
def maps_link(addresses):
    clean = [str(a) for a in addresses if pd.notna(a)]
    return "https://www.google.com/maps/dir/" + "/".join([quote(a) for a in clean])


# ---------------- MAIN ----------------
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

    # ---------------- GEOCODE ----------------
    lat, lon = [], []

    for _, row in df.iterrows():
        la, lo = geocode_postcode(row["postcode"])
        lat.append(la)
        lon.append(lo)

    df["lat"] = lat
    df["lon"] = lon

    df = df.dropna(subset=["lat", "lon"]).reset_index(drop=True)

    if len(df) == 0:
        st.error("No valid geocoded jobs.")
        st.stop()

    st.success(f"Valid jobs: {len(df)}")

    # =========================================================
    # 🧭 GEOGRAPHIC CLUSTERING (KEY FIX)
    # =========================================================

    kmeans = KMeans(n_clusters=engineers, random_state=42, n_init="auto")
    df["Engineer"] = kmeans.fit_predict(df[["lat", "lon"]])

    # =========================================================
    # 📦 OUTPUT
    # =========================================================

    for e in range(engineers):

        eng_df = df[df["Engineer"] == e].copy()

        st.subheader(f"Engineer {e+1}")

        # route optimised per engineer
        eng_df = order_route(eng_df)

        # show everything clearly including AM/PM
        cols = [address_col, "postcode"]
        if "Slot" in eng_df.columns:
            cols.append("Slot")
        cols += ["lat", "lon"]

        st.dataframe(eng_df[cols])

        if not eng_df.empty:
            link = maps_link(eng_df[address_col].tolist())
            st.markdown(f"[Open Route in Google Maps]({link})")
