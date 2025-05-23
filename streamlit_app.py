# partBPlaceholder/streamlit_app.py

import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium

from core import load_timeseries, build_weighted_graph, k_shortest_paths, MODEL_MAP

st.set_page_config(page_title="Part B Path Explorer", layout="wide")
st.sidebar.title("Part B Path Explorer")

# load dropdown options once
nodes_df = pd.read_csv("data/scats_complete_average.csv")
nodes_df["Site_ID"] = nodes_df["Site_ID"].astype(str)
nodes_df["label"]   = nodes_df["Site_ID"] + " ⎯ " + nodes_df["Location"]

# --- sidebar inputs ---
start_sel = st.sidebar.selectbox("Start node", nodes_df["label"])
end_sel   = st.sidebar.selectbox("End node",   nodes_df["label"])
ts_str    = st.sidebar.text_input("Timestamp (YYYY-MM-DD HH:MM:SS)", "2006-10-02 05:15:00")
model_name= st.sidebar.selectbox("Prediction model", list(MODEL_MAP.keys()), index=1)
n_paths   = st.sidebar.number_input("Number of paths", 1, 10, 3)

# parse timestamp
try:
    timestamp = pd.to_datetime(ts_str)
except Exception:
    st.sidebar.error("Invalid timestamp—use YYYY-MM-DD HH:MM:SS")
    st.stop()

# initialize our session_state slots
if "G"        not in st.session_state: st.session_state.G = None
if "paths"    not in st.session_state: st.session_state.paths = None
if "params"   not in st.session_state: st.session_state.params = {}

# when you click, compute & stash everything
if st.sidebar.button("Show paths"):
    params = {
      "start": start_sel,
      "end":   end_sel,
      "ts":    timestamp,
      "model": model_name,
      "k":     n_paths
    }
    # only recompute if you’ve changed any parameter
    if params != st.session_state.params:
        st.session_state.params = params.copy()
        with st.spinner("Building graph & predicting…"):
            ts_df = load_timeseries()
            source = start_sel.split("⎯")[0].strip()
            target = end_sel.split("⎯")[0].strip()

            G = build_weighted_graph(ts_df, model_name, timestamp)
            paths = k_shortest_paths(G, source, target, n_paths)

        st.session_state.G     = G
        st.session_state.paths = paths

PALETTE = [
    "red", "blue", "green", "orange", "purple",
    "darkred", "cadetblue", "darkgreen", "darkblue", "brown"
]

# now, if we have st.session_state.paths, draw them (and only draw, no recompute)
if st.session_state.paths:
    G     = st.session_state.G
    paths = st.session_state.paths

    if not paths:
        st.error("No paths found.")
    else:
        # 1) let the user pick which path to highlight
        path_choices = [
            f"{i}. {time:.1f} min, {dist:.2f} km →  {' → '.join(nodes)}"
            for i, (nodes, time, dist) in enumerate(paths, start=1)]        
        # default to the first path if none chosen yet
        if "highlight" not in st.session_state:
            st.session_state.highlight = path_choices[0]

        st.sidebar.radio(
            "Bring path to front:",
            path_choices,
            key="highlight",
        )

        # figure out which index that is
        highlight_idx = int(st.session_state.highlight.split(".")[0]) - 1
        lat, lon = G.nodes[
            st.session_state.params["start"].split("⎯")[0].strip()
        ]["pos"]
        m = folium.Map(location=[lat, lon], zoom_start=13)
        all_coords = []
        for idx, (nodes, total_time, total_dist) in enumerate(paths, start=1):
            colour = PALETTE[(idx-1) % len(PALETTE)]
            coords = [ G.nodes[n]["pos"] for n in nodes ]
            all_coords.extend(coords)
            folium.PolyLine(
                coords,
                color=colour,
                weight=4,
                tooltip=f"Path {idx}: {total_time:.1f} min, {total_dist:.2f} km"
            ).add_to(m)

        # 2) auto-center/zoom so every path is visible
        lats = [lat for lat,lon in all_coords]
        lons = [lon for lat,lon in all_coords]
        m.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])

        # 3) display the map
        st_folium(m, width=800, height=500)

        # 4) render the textual list
        st.subheader("Paths")
        for idx, (nodes, total_time, total_dist) in enumerate(paths, start=1):
            colour = PALETTE[(idx-1) % len(PALETTE)]
            path_str = " → ".join(nodes)
            st.markdown(
                f"<span style='color:{colour}'>"
                f"**{idx}.** {total_time:.1f} min, {total_dist:.2f} km — {path_str}"
                f"</span>",
                unsafe_allow_html=True
            )