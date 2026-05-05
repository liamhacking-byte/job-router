import streamlit as st
import pandas as pd
import numpy as np
import requests
from urllib.parse import quote
from math import radians, sin, cos, sqrt, atan2

from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2

st.title("Smart Job Router (Optimised Dispatch Engine - OR Tools)")

uploaded_file = st.file_uploader("Upload Jobs Excel")
engineers = st.number_input("Number of engineers", min_value=1, value=4)

# ---------------- SETTINGS ----------------
MIN_JOBS = 4
MAX_JOBS = 5


# ---------------- POSTCODE EXTRACTION ----------------
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


# ---------------- DISTANCE (HAVERSINE) ----------------
def distance(a, b):
    R = 6371
    lat1, lon1 = map(radians, a)
    lat2, lon2 = map(radians, b)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    h = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    return 2 * R * atan2(sqrt(h), sqrt(1 - h))


# ---------------- GOOGLE MAPS ----------------
def maps_link(addresses):
    clean = [str(a).strip() for a in addresses if pd.notna(a)]

    if len(clean) <= 1:
        return None

    return (
        "https://www.google.com/maps/dir/?api=1"
        "&travelmode=driving"
        "&waypoints=" + "|".join(quote(a) for a in clean[1:])
    )


# ---------------- OR-TOOLS SOLVER ----------------
def solve_vrp(distance_matrix, num_vehicles, demands, vehicle_capacity):

    manager = pywrapcp.RoutingIndexManager(
        len(distance_matrix),
        num_vehicles,
        0  # depot = first node
    )

    routing = pywrapcp.RoutingModel(manager)

    # distance callback
    def dist_cb(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return distance_matrix[from_node][to_node]

    transit_cb = routing.RegisterTransitCallback(dist_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_cb)

    # capacity constraint
    def demand_cb(from_index):
        from_node = manager.IndexToNode(from_index)
        return demands[from_node]

    demand_cb_idx = routing.RegisterUnaryTransitCallback(demand_cb)

    routing.AddDimensionWithVehicleCapacity(
        demand_cb_idx,
        0,
        vehicle_capacity,
        True,
        "Capacity"
    )

    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )

    solution = routing.SolveWithParameters(search_params)

    if not solution:
        return None

    routes = []

    for vehicle_id in range(num_vehicles):

        index = routing.Start(vehicle_id)
        route = []

        while not routing.IsEnd(index):

            node = manager.IndexToNode(index)
            route.append(node)
            index = solution.Value(routing.NextVar(index))

        routes.append(route)

    return routes


# ---------------- MAIN APP ----------------
if uploaded_file:

    df = pd.read_excel(uploaded_file)
    df.columns = df.columns.str.strip()

    address_col = next((c for c in df.columns if "address" in c.lower()), None)

    if not address_col:
        st.error("No address column found.")
        st.stop()

    st.write("Using column:", address_col)

    # ---------------- POSTCODES ----------------
    df["postcode"] = df[address_col].apply(extract_postcode)

    # ---------------- GEOCODING ----------------
    unique_postcodes = df["postcode"].dropna().unique()

    cache = {}
    for pc in unique_postcodes:
        cache[pc] = geocode_postcode(pc)

    df["lat"] = df["postcode"].map(lambda x: cache.get(x, (None, None))[0])
    df["lon"] = df["postcode"].map(lambda x: cache.get(x, (None, None))[1])

    df = df.dropna(subset=["lat", "lon"]).reset_index(drop=True)

    if df.empty:
        st.error("No valid geocoded jobs.")
        st.stop()

    st.success(f"Valid jobs: {len(df)}")

    # ======================================================
    # 🧠 BUILD DISTANCE MATRIX
    # ======================================================

    coords = list(zip(df["lat"], df["lon"]))

    n = len(coords)

    distance_matrix = [
        [
            int(distance(coords[i], coords[j]) * 1000)
            for j in range(n)
        ]
        for i in range(n)
    ]

    # demand = 1 per job
    demands = [1] * n

    # vehicle capacity = 4–5 jobs
    capacity = MAX_JOBS

    # ======================================================
    # 🚀 SOLVE VRP
    # ======================================================

    routes = solve_vrp(
        distance_matrix,
        engineers,
        demands,
        capacity
    )

    if not routes:
        st.error("No solution found.")
        st.stop()

    st.success("Optimised routing complete!")

    # ======================================================
    # 📦 OUTPUT
    # ======================================================

    for i, route in enumerate(routes):

        st.subheader(f"Engineer {i + 1}")

        if len(route) <= 1:
            continue

        eng_df = df.iloc[route].copy()

        st.dataframe(
            eng_df[[address_col, "postcode", "lat", "lon"]]
        )

        link = maps_link(eng_df[address_col].tolist())

        if link:
            st.markdown(f"[Open Route in Google Maps]({link})")
