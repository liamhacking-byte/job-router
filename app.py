import streamlit as st
import pandas as pd
import numpy as np
import requests
from urllib.parse import quote

st.title("Smart Job Router (Stable Version)")

uploaded_file = st.file_uploader("Upload Jobs Excel")

ENGINEERS = st.number_input("Engineers", min_value=1, value=4)
MAX_JOBS = 5


# ---------------- POSTCODE ----------------
def extract_postcode(text):
    if pd.isna(text):
        return None
    return str(text).split(",")[-1].strip().upper()


# ---------------- GEOCODE ----------------
def geocode(pc):
    try:
        url = f"https://api.postcodes.io/postcodes/{pc.replace(' ', '')}"
        r = requests.get(url, timeout=10).json()
        if r["status"] != 200:
            return None, None
        return r["result"]["latitude"], r["result"]["longitude"]
    except:
        return None, None


# ---------------- DISTANCE ----------------
def dist(a, b):
    return (a[0]-b[0])**2 + (a[1]-b[1])**2


# ---------------- ROUTE ORDER ----------------
def route(df):
    if len(df) <= 1:
        return df

    remaining = df.copy()
    out = []

    current = remaining.iloc[0]
    out.append(current)
    remaining = remaining.drop(current.name)

    while len(remaining) > 0:
        cp = (current["lat"], current["lon"])

        remaining["d"] = remaining.apply(
            lambda r: dist(cp, (r["lat"], r["lon"])),
            axis=1
        )

        nxt = remaining["d"].idxmin()
        current = remaining.loc[nxt]
        out.append(current)
        remaining = remaining.drop(nxt)

    return pd.DataFrame(out).drop(columns=["d"], errors="ignore")


# ---------------- MAP ----------------
def map_link(addrs):
    return "https://www.google.com/maps/dir/" + "/".join([quote(str(a)) for a in addrs])


# ---------------- MAIN ----------------
if uploaded_file:

    df = pd.read_excel(uploaded_file)
    df.columns = df.columns.str.strip()

    address_col = next((c for c in df.columns if "address" in c.lower()), None)

    if not address_col:
        st.error("No address column found")
        st.stop()

    # ---------------- POSTCODES ----------------
    df["postcode"] = df[address_col].apply(extract_postcode)

    lat, lon = [], []

    for pc in df["postcode"]:
        la, lo = geocode(pc)
        lat.append(la)
        lon.append(lo)

    df["lat"] = lat
    df["lon"] = lon

    df = df.dropna(subset=["lat", "lon"]).reset_index(drop=True)

    st.write(f"Valid jobs: {len(df)}")

    # =====================================================
    # 🧠 CORE LOGIC: NEAREST ENGINEER + CAPACITY LIMIT
    # =====================================================

    df["Engineer"] = -1
    counts = {i: 0 for i in range(ENGINEERS)}

    # seed engineers with first points (spread start)
    seeds = df.sample(min(ENGINEERS, len(df)), random_state=1)
    centers = list(zip(seeds["lat"], seeds["lon"]))

    if len(centers) < ENGINEERS:
        centers += centers[:ENGINEERS - len(centers)]

    for i, row in df.iterrows():

        point = (row["lat"], row["lon"])

        best = None
        best_score = 1e9

        for e in range(ENGINEERS):

            if counts[e] >= MAX_JOBS:
                continue

            score = dist(point, centers[e]) + counts[e] * 0.1

            if score < best_score:
                best_score = score
                best = e

        if best is None:
            best = min(counts, key=counts.get)

        df.at[i, "Engineer"] = best
        counts[best] += 1

    st.success("Routing complete")

    # =====================================================
    # OUTPUT
    # =====================================================

    for e in range(ENGINEERS):

        eng = df[df["Engineer"] == e].copy()

        st.subheader(f"Engineer {e+1}")

        eng = route(eng)

        cols = [address_col, "postcode"]
        if "Slot" in eng.columns:
            cols.append("Slot")
        cols += ["lat", "lon"]

        st.dataframe(eng[cols])

        if len(eng) > 0:
            st.markdown(
                f"[Open Route]({map_link(eng[address_col].tolist())})"
            )
