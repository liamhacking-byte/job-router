import streamlit as st
import pandas as pd
import numpy as np
import requests
from urllib.parse import quote

st.title("Smart Job Router (Dispatch + AM/PM)")

uploaded_file = st.file_uploader("Upload Jobs Excel")
engineers = st.number_input("Number of engineers", min_value=1, value=4)


# ---------------- POSTCODE EXTRACTION ----------------
def extract_postcode(text):
    if pd.isna(text):
        return None

    parts = [p.strip() for p in str(text).split(",") if p.strip()]
    if not parts:
        return None

    return parts[-1].upper()


# ---------------- UK POSTCODE GEOCODING ----------------
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


# ---------------- DISTANCE ----------------
def dist(a, b):
    return (a[0] - b[0])**2 + (a[1] - b[1])**2


# ---------------- ROUTE ORDER (nearest neighbour) ----------------
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

        remaining["d"] = remaining.apply(
            lambda r: dist(current_point, (r["lat"], r["lon"])),
            axis=1
        )

        nxt = remaining["d"].idxmin()
        current = remaining.loc[nxt]

        route.append(current)
        remaining = remaining.drop(nxt)

    return pd.DataFrame(route).drop(columns=["d"], errors="ignore")


# ---------------- MAP LINK ----------------
def maps_link(addresses):
    clean = [str(a) for a in addresses if pd.notna(a)]
    return "https://www.google.com/maps/dir/" + "/".join([quote(a) for a in clean])


# ---------------- MAIN APP ----------------
if uploaded_file:

    df = pd.read_excel(uploaded_file)
    df.columns = df.columns.str.strip()

    address_col = next((c for c in df.columns if "address" in c.lower()), None)

    if not address_col:
        st.error(f"No address column found. Columns: {list(df.columns)}")
        st.stop()

    st.write("Using:", address_col)

    # ---------------- POSTCODES ----------------
    df["postcode"] = df[address_col].apply(extract_postcode)

    st.write("Sample postcodes:")
    st.write(df["postcode"].head(10))

    # ---------------- GEOCODING ----------------
    lat_list = []
    lon_list = []
    failed = []

    for _, row in df.iterrows():

        lat, lon = geocode_postcode(row["postcode"])

        if lat is None:
            failed.append(True)

        lat_list.append(lat)
        lon_list.append(lon)

    df["lat"] = lat_list
    df["lon"] = lon_list

    st.warning(f"Failed geocoding: {len(failed)} jobs")

    df = df.dropna(subset=["lat", "lon"]).reset_index(drop=True)

    if len(df) == 0:
        st.error("No valid jobs after geocoding.")
        st.stop()

    st.write(f"Valid jobs: {len(df)}")

    # =========================================================
    # GEOGRAPHIC SEED CLUSTERING (BALANCED)
    # =========================================================

    df = df.reset_index(drop=True)

    # pick spread-out seeds
    seed_idx = np.linspace(0, len(df) - 1, engineers, dtype=int)
    seeds = df.iloc[seed_idx]

    centers = list(zip(seeds["lat"], seeds["lon"]))

    df["Engineer"] = -1

    # assign nearest seed
    for i, row in df.iterrows():

        p = (row["lat"], row["lon"])

        dists = [(e, dist(p, c)) for e, c in enumerate(centers)]
        df.at[i, "Engineer"] = min(dists, key=lambda x: x[1])[0]

    # balance workloads gently
    max_jobs = int(np.ceil(len(df) / engineers))

    for e in range(engineers):

        cluster = df[df["Engineer"] == e]

        if len(cluster) > max_jobs:

            excess = cluster.iloc[max_jobs:]

            for idx, row in excess.iterrows():

                p = (row["lat"], row["lon"])

                options = []

                for j in range(engineers):
                    if len(df[df["Engineer"] == j]) < max_jobs:
                        options.append((j, dist(p, centers[j])))

                if options:
                    new_e = min(options, key=lambda x: x[1])[0]
                    df.at[idx, "Engineer"] = new_e

    st.success("Routing complete!")

    # =========================================================
    # OUTPUT (AM / PM RESTORED)
    # =========================================================

    for e in range(engineers):

        eng_df = df[df["Engineer"] == e].copy()

        st.subheader(f"Engineer {e+1}")

        if "Slot" in eng_df.columns:

            am = eng_df[eng_df["Slot"] == "AM"].copy()
            pm = eng_df[eng_df["Slot"] == "PM"].copy()

            am = order_route(am) if len(am) > 1 else am
            pm = order_route(pm) if len(pm) > 1 else pm

            route = pd.concat([am, pm])

        else:
            route = order_route(eng_df)

        st.dataframe(route[[address_col, "postcode", "lat", "lon"]])

        if not route.empty:
            link = maps_link(route[address_col].tolist())
            st.markdown(f"[Open Route in Google Maps]({link})")
