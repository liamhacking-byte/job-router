import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.cluster import KMeans
from urllib.parse import quote

st.title("Smart Job Router (Stable Geo Clustering)")

uploaded_file = st.file_uploader("Upload Jobs Excel")

engineers = st.number_input("Number of engineers", min_value=1, value=4)
MAX_JOBS = 5


# ---------------- POSTCODE ----------------
def extract_postcode(text):
    if pd.isna(text):
        return None
    parts = str(text).split(",")
    return parts[-1].strip().upper()


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
            lambda r: (r["lat"] - cp[0])**2 + (r["lon"] - cp[1])**2,
            axis=1
        )

        nxt = remaining["d"].idxmin()
        current = remaining.loc[nxt]
        out.append(current)
        remaining = remaining.drop(nxt)

    return pd.DataFrame(out).drop(columns=["d"], errors="ignore")


# ---------------- MAP ----------------
def maps(addrs):
    return "https://www.google.com/maps/dir/" + "/".join([quote(str(a)) for a in addrs])


# ---------------- MAIN ----------------
if uploaded_file:

    df = pd.read_excel(uploaded_file)
    df.columns = df.columns.str.strip()

    address_col = [c for c in df.columns if "address" in c.lower()][0]

    df["postcode"] = df[address_col].apply(extract_postcode)

    lat, lon = [], []

    for p in df["postcode"]:
        la, lo = geocode(p)
        lat.append(la)
        lon.append(lo)

    df["lat"] = lat
    df["lon"] = lon

    df = df.dropna(subset=["lat", "lon"]).reset_index(drop=True)

    st.write(f"Valid jobs: {len(df)}")

    # =====================================================
    # 🧭 STEP 1: GEO CLUSTERING (CLEAN & STABLE)
    # =====================================================

    n_clusters = engineers * MAX_JOBS

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    df["cluster"] = kmeans.fit_predict(df[["lat", "lon"]])

    # =====================================================
    # 🧠 STEP 2: ASSIGN CLUSTERS EVENLY
    # =====================================================

    cluster_sizes = df["cluster"].value_counts().sort_values(ascending=False)

    engineer_load = {i: 0 for i in range(engineers)}
    cluster_map = {}

    for c, size in cluster_sizes.items():

        best = min(engineer_load, key=engineer_load.get)

        if engineer_load[best] + size <= MAX_JOBS:
            cluster_map[c] = best
            engineer_load[best] += size
        else:
            # force assign if needed
            cluster_map[c] = best
            engineer_load[best] += min(size, MAX_JOBS)

    df["Engineer"] = df["cluster"].map(cluster_map)

    st.success("Routing complete")

    # =====================================================
    # OUTPUT
    # =====================================================

    for e in range(engineers):

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
                f"[Open Route](https://www.google.com/maps/dir/" +
                "/".join([quote(str(x)) for x in eng[address_col]]) +
                ")"
            )
