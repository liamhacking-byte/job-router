import streamlit as st
import pandas as pd
import numpy as np
import requests
from urllib.parse import quote

st.title("Smart Job Router (Geo-Optimised)")

uploaded_file = st.file_uploader("Upload Jobs Excel")
engineers = st.number_input("Number of engineers", min_value=1, value=4)


# ---------------- POSTCODE EXTRACTION ----------------
def extract_postcode(text):
    if pd.isna(text):
        return None
    parts = [p.strip() for p in str(text).split(",") if p.strip()]
    return parts[-1].upper() if parts else None


# ---------------- GEOCODE ----------------
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


# ---------------- DISTANCE ----------------
def dist(a, b):
    return (a[0] - b[0])**2 + (a[1] - b[1])**2


# ---------------- ROUTE ORDER ----------------
def order_route(df):
    if len(df) <= 1:
        return df

    remaining = df.copy()
    route = []

    current = remaining.iloc[0]
    route.append(current)
    remaining = remaining.drop(current.name)

    while len(remaining) > 0:

        cp = (current["lat"], current["lon"])

        remaining["d"] = remaining.apply(
            lambda r: dist(cp, (r["lat"], r["lon"])),
            axis=1
        )

        nxt = remaining["d"].idxmin()
        current = remaining.loc[nxt]

        route.append(current)
        remaining = remaining.drop(nxt)

    return pd.DataFrame(route).drop(columns=["d"], errors="ignore")


# ---------------- MAP ----------------
def maps_link(addrs):
    clean = [str(a) for a in addrs if pd.notna(a)]
    return "https://www.google.com/maps/dir/" + "/".join([quote(a) for a in clean])


# ---------------- MAIN ----------------
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

    # ---------------- GEOCODE ----------------
    lat, lon = [], []

    for _, row in df.iterrows():
        la, lo = geocode_postcode(row["postcode"])
        lat.append(la)
        lon.append(lo)

    df["lat"] = lat
    df["lon"] = lon

    df = df.dropna(subset=["lat", "lon"]).reset_index(drop=True)

    st.success(f"Valid jobs: {len(df)}")

    # =========================================================
    # 🧭 GEO-FIRST BALANCED ASSIGNMENT (NO AM/PM LOGIC)
    # =========================================================

    df["Engineer"] = -1
    counts = {i: 0 for i in range(engineers)}
    target = len(df) / engineers

    for i, row in df.iterrows():

        point = (row["lat"], row["lon"])

        best_engineer = None
        best_score = float("inf")

        for e in range(engineers):

            eng_jobs = df[df["Engineer"] == e]

            if len(eng_jobs) == 0:
                centroid = point
            else:
                centroid = (eng_jobs["lat"].mean(), eng_jobs["lon"].mean())

            geo = dist(point, centroid)

            # light balancing only
            load_penalty = abs(counts[e] - target) * 0.2

            score = geo + load_penalty

            if score < best_score:
                best_score = score
                best_engineer = e

        df.at[i, "Engineer"] = best_engineer
        counts[best_engineer] += 1

    st.success("Routing complete!")

    # =========================================================
    # 📦 OUTPUT (AM/PM DISPLAY ONLY)
    # =========================================================

    for e in range(engineers):

        eng_df = df[df["Engineer"] == e].copy()

        st.subheader(f"Engineer {e+1}")

        eng_df = order_route(eng_df)

        cols = [address_col, "postcode"]

        # AM/PM is ONLY shown, not used in routing
        if "Slot" in eng_df.columns:
            cols.append("Slot")

        cols += ["lat", "lon"]

        st.dataframe(eng_df[cols])

        if not eng_df.empty:
            link = maps_link(eng_df[address_col].tolist())
            st.markdown(f"[Open Route in Google Maps]({link})")
