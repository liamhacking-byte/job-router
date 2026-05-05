import streamlit as st
import pandas as pd
import numpy as np
import requests
from urllib.parse import quote
from math import radians, sin, cos, sqrt, atan2

from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2

st.title("Smart Job Router (Fast Optimised Dispatch Engine)")

uploaded_file = st.file_uploader("Upload Jobs Excel")
engineers = st.number_input("Number of engineers", min_value=1, value=4)

# ---------------- SETTINGS ----------------
MAX_JOBS = 5
MIN_JOBS = 4
MAX_NODES = 120  # performance cap


# ---------------- POSTCODE ----------------
def extract_postcode(text):
    if pd.isna(text):
        return None
    parts = [p.strip() for p in str(text).split(",") if p.strip()]
    return parts[-1].upper() if parts else None


# ---------------- GEOCODING ----------------
@st.cache_data
def geocode_postcode(postcode):
    if not postcode:
        return None, None

    try:
        url = f"https://api.postcodes.io/postcodes/{postcode.replace(' ', '')}"
        r = requests.get(url, timeout=10).json()

        if r["status"] != 200:
            return None, None

        res = r["result"]
        return res["latitude"], res["longitude"]

    except:
        return None, None


# ---------------- DISTANCE ----------------
def distance(a, b):
    R = 6371
    lat1, lon1 = map(radians, a)
    lat2, lon2 = map(radians, b)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    h = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    return 2 * R * atan2(sqrt(h), sqrt(1 - h))


# ---------------- MAPS ----------------
def maps_link(addresses):
    clean = [str(a).strip() for a in addresses if pd.notna(a)]

    if len(clean) <= 1:
        return None

    return (
        "https://www.google.com/maps/dir/?api=1"
        "&travelmode=driving"
        "&waypoints=" + "|".join(quote(a) for a in clean[1:])
    )


# ---------------- DISTANCE MATRIX (CACHED) ----------------
@st.cache_data
def build_matrix(coords):
    n = len(coords)

    matrix = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(int(distance(coords[i], coords[j]) * 1000))
        matrix.append(row)

    return matrix


# ---------------- OR-TOOLS SOLVER ----------------
def solve_vrp(distance_matrix, num_vehicles, demands, capacities):

    manager = pywrapcp.RoutingIndexManager(
        len(distance_matrix),
        num_vehicles,
        0
    )

    routing = pywrapcp.RoutingModel(manager)

    # distance callback
    def dist_cb(from_index, to_index):
        return distance_matrix[
            manager.IndexToNode(from_index)
        ][
            manager.IndexToNode(to_index)
        ]

    transit_cb = routing.RegisterTransitCallback(dist_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_cb)

    # demand callback
    def demand_cb(from_index):
        return demands[manager.IndexToNode(from_index)]

    demand_cb_idx = routing.RegisterUnaryTransitCallback(demand_cb)

    routing.AddDimensionWithVehicleCapacity(
        demand_cb_idx,
        0,
        capacities,
        True,
        "Capacity"
    )

    # ⚡ FAST SOLVER SETTINGS (IMPORTANT FIX)
    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )

    search_params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )

    search_params.time_limit.FromSeconds(3)

    solution = routing.SolveWithParameters(search_params)

    if not solution:
        return None

    routes = []

    for v in range(num_vehicles):

        index = routing.Start(v)
        route = []

        while not routing.IsEnd(index):

            route.append(manager.IndexToNode(index))
            index = solution.Value(routing.NextVar(index))

        routes.append(route)

    return routes


# ---------------- MAIN ----------------
if uploaded_file:

    df = pd.read_excel(uploaded_file)
    df.columns = df.columns.str.strip()

    address_col = next((c for c in df.columns if "address" in c.lower()), None)

    if not address_col:
        st.error("No address column found")
        st.stop()

    st.write("Using column:", address_col)

    # ---------------- LIMIT DATA FOR SPEED ----------------
    if len(df) > MAX_NODES:
        st.warning(f"Large dataset detected. Limiting to {MAX_NODES} jobs for performance.")
        df = df.sample(MAX_NODES, random_state=42)

    # ---------------- POSTCODES ----------------
    df["postcode"] = df[address_col].apply(extract_postcode)

    # ---------------- GEOCODE ----------------
    unique_pcs = df["postcode"].dropna().unique()

    cache = {pc: geocode_postcode(pc) for pc in unique_pcs}

    df["lat"] = df["postcode"].map(lambda x: cache.get(x, (None, None))[0])
    df["lon"] = df["postcode"].map(lambda x: cache.get(x, (None, None))[1])

    df = df.dropna(subset=["lat", "lon"]).reset_index(drop=True)

    if df.empty:
        st.error("No valid geocoded jobs")
        st.stop()

    st.success(f"Valid jobs: {len(df)}")

    # ---------------- PREP DATA ----------------
    coords = list(zip(df["lat"], df["lon"]))

    distance_matrix = build_matrix(coords)

    demands = [1] * len(df)

    capacities = [MAX_JOBS] * engineers

    # safety check
    if len(df) > engineers * MAX_JOBS:
        st.error("Too many jobs for available engineers (max 5 each).")
        st.stop()

    # ---------------- SOLVE ----------------
    with st.spinner("Optimising routes..."):
        routes = solve_vrp(
            distance_matrix,
            engineers,
            demands,
            capacities
        )

    if not routes:
        st.error("No solution found")
        st.stop()

    st.success("Routing complete!")

    # ---------------- OUTPUT ----------------
    for i, route in enumerate(routes):

        st.subheader(f"Engineer {i + 1}")

        if len(route) <= 1:
            continue

        eng_df = df.iloc[route]

        st.dataframe(
            eng_df[[address_col, "postcode", "lat", "lon"]]
        )

        link = maps_link(eng_df[address_col].tolist())

        if link:
            st.markdown(f"[Open Route in Google Maps]({link})")
