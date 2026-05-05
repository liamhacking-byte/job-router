import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.cluster import KMeans
from urllib.parse import quote

st.title("Smart Job Router (Stable Geo Version)")

uploaded_file = st.file_uploader("Upload Jobs Excel")

ENGINEERS = st.number_input("Engineers", min_value=1, value=4)
MAX_JOBS = 5


# ---------------- POSTCODE ----------------
def extract_postcode(x):
    if pd.isna(x):
        return None
    return str(x).split(",")[-1].strip().upper()


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


# ---------------- ROUTE ----------------
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
            lambda r: (r["lat"] - cp[0])**2 + (r["lon"] - cp[1])**2,
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
    # 🧭 STEP 1: GEO CLUSTERING
    # =====================================================

    n_clusters = ENGINEERS * MAX_JOBS

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    df["cluster"] = kmeans.fit_predict(df[["lat", "lon"]])

    # sort clusters by size (biggest first)
    cluster_order = df["cluster"].value_counts().index.tolist()

    # =====================================================
    # 🧠 STEP 2: SIMPLE BALANCED ASSIGNMENT
    # =====================================================

    df["Engineer"] = -1
    counts = {i: 0 for i in range(ENGINEERS)}

    for c in cluster_order:

        cluster_df = df[df["cluster"] == c]

        for _, row in cluster_df.iterrows():

            best = min(counts, key=counts.get)

            if counts[best] >= MAX_JOBS:
                continue

            df.loc[row.name, "Engineer"] = best
            counts[best] += 1

    df = df[df["Engineer"] != -1]

    st.success("Routing complete")

    # =====================================================
    # OUTPUT
    # =====================================================

    for e in range(ENGINEERS):

        eng = df[df["Engineer"] == e].copy()

        st.subheader(f"Engineer {e+1}")

        eng = route(eng)

        st.dataframe(eng[[address_col, "postcode", "lat", "lon"]])

        if len(eng):
            st.markdown(
                f"[Open Route]({map_link(eng[address_col].tolist())})"
            )
