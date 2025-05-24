# # partBPlaceholder/streamlit_app.py

# import streamlit as st
# import pandas as pd
# import folium
# from streamlit_folium import st_folium

# from core import load_timeseries, build_weighted_graph, k_shortest_paths, MODEL_MAP

# st.set_page_config(page_title="Part B Path Explorer", layout="wide")
# st.sidebar.title("Part B Path Explorer")

# if "G"      not in st.session_state: st.session_state.G      = None
# if "paths"  not in st.session_state: st.session_state.paths  = None
# if "params" not in st.session_state: st.session_state.params = {}

# # load dropdown options once
# nodes_df = pd.read_csv("data/scats_complete_average.csv")
# nodes_df["Site_ID"] = nodes_df["Site_ID"].astype(str)
# nodes_df["label"]   = nodes_df["Site_ID"] + " ⎯ " + nodes_df["Location"]
# nodes_df["Location"] = nodes_df["Location"].str.replace("/", " / ", regex=False)

# # --- sidebar inputs ---
# start_sel = st.sidebar.selectbox("Start node", nodes_df["label"], index=0)
# end_sel   = st.sidebar.selectbox("End node",   nodes_df["label"], index=1)
# # ts_str    = st.sidebar.text_input("Timestamp (YYYY-MM-DD HH:MM:SS)", "2006-10-05 05:15:00")


# # # parse timestamp
# # try:
# #     timestamp = pd.to_datetime(ts_str)
# # except Exception:
# #     st.sidebar.error("Invalid timestamp—use YYYY-MM-DD HH:MM:SS")
# #     st.stop()

# with st.sidebar:
#     st.markdown("### Date (October 2006)")
#     col_day, col_hour, col_min = st.columns(3)

#     day = col_day.selectbox(
#         "Day",
#         options=list(range(1, 32)),
#         index=4,
#         key="day"
#     )

#     hour = col_hour.selectbox(
#         "Hour",
#         options=list(range(0, 24)),
#         index=5,
#         key="hour"
#     )

#     minute = col_min.selectbox(
#         "Min",
#         options=[0, 15, 30, 45],
#         index=1,
#         key="minute"
#     )

# # now construct your timestamp
# try:
#     timestamp = pd.Timestamp(year=2006, month=10,
#                              day=day, hour=hour,
#                              minute=minute, second=0)
# except ValueError as e:
#     st.sidebar.error(f"Invalid date/time: {e}")
#     st.stop()

# model_name= st.sidebar.selectbox("Prediction model", list(MODEL_MAP.keys()), index=0)
# n_paths   = st.sidebar.number_input("Number of paths", 1, 10, 3)

# # initialize our session_state slots
# if "G"        not in st.session_state: st.session_state.G = None
# if "paths"    not in st.session_state: st.session_state.paths = None
# if "params"   not in st.session_state: st.session_state.params = {}

# # when you click, compute & stash everything
# if st.sidebar.button("Show paths"):
#     params = {
#       "start": start_sel,
#       "end":   end_sel,
#       "ts":    timestamp,
#       "model": model_name,
#       "k":     n_paths
#     }
#     # only recompute if you’ve changed any parameter
#     if params != st.session_state.params:
#         st.session_state.params = params.copy()
#         with st.spinner("Building graph & predicting…"):
#             ts_df = load_timeseries()
#             source = start_sel.split("⎯")[0].strip()
#             target = end_sel.split("⎯")[0].strip()

#             G = build_weighted_graph(ts_df, model_name, timestamp)
#             paths = k_shortest_paths(G, source, target, n_paths)

#         st.session_state.G     = G
#         st.session_state.paths = paths

# PALETTE = [
#     "red", "blue", "green", "purple", "darkred",
#     "cadetblue", "darkgreen", "darkblue", "brown"
# ]

# # now, if we have st.session_state.paths, draw them (and only draw, no recompute)
# if st.session_state.paths:
#     G     = st.session_state.G
#     paths = st.session_state.paths

#     if not paths:
#         st.error("No paths found.")
#     else:
#         # 1) let the user pick which path to highlight
#         path_choices = [
#             f"{i}. {time:.1f} min, {dist:.2f} km →  {' → '.join(nodes)}"
#             for i, (nodes, time, dist) in enumerate(paths, start=1)]        
#         # default to the first path if none chosen yet
#         if "highlight" not in st.session_state:
#             st.session_state.highlight = path_choices[0]
#         # st.sidebar.radio("Bring path to front:", path_choices, key="highlight")
#         # default to "All nodes" if no paths yet, else default to highlighted path
#         default_idx = 2 if st.session_state.paths else 0
#         if len(path_choices) == 1:
#             node_display_option = st.sidebar.radio(
#                 "Nodes to display:",
#                 ["All nodes", "Highlighted path nodes"],
#                 index=default_idx -1
#             )
#         else:
#             node_display_option = st.sidebar.radio(
#                 "Nodes to display:",
#                 ["All nodes", "All path nodes", "Highlighted path nodes"],
#                 index=default_idx
#             )
#             st.sidebar.radio("Bring path to front:", path_choices, key="highlight")

#         # figure out which index that is
#         highlight_idx = int(st.session_state.highlight.split(".")[0]) - 1

#         if node_display_option == "All nodes":
#             nodes_to_show = list(G.graph["centroids"].keys())
#         elif node_display_option == "All path nodes":
#             # union of every path’s node list
#             nodes_to_show = sorted({
#                 n for (nodes, _, _) in paths for n in nodes
#             })
#         else:  # “Highlighted path nodes”
#             # only nodes in the highlighted route
#             _, highlighted_nodes, _ = None, [], None
#             # unpack the highlighted tuple
#             hl = paths[highlight_idx]
#             highlighted_nodes = hl[0]
#             nodes_to_show = highlighted_nodes
#         lat, lon = G.nodes[st.session_state.params["start"].split("⎯")[0].strip()]["pos"]
#         m = folium.Map(location=[lat, lon], zoom_start=13)
#         for node_id in nodes_to_show:
#             lat, lon = G.nodes[node_id]["pos"]
#             location = nodes_df.loc[nodes_df["Site_ID"] == str(node_id), "Location"].values[0]
#             folium.Marker(
#                 location=(lat, lon),
#                 popup=f"Node {node_id}, {location}  ({lat:.5f}, {lon:.5f})",
#                 icon=folium.Icon(color="gray", icon="info-sign"),
#             ).add_to(m)

#         all_coords = []
#         for idx, (nodes, total_time, total_dist) in enumerate(paths):
#             coords = [ G.nodes[n]["pos"] for n in nodes ]
#             all_coords.extend(coords)
#             if idx == highlight_idx:
#                 colour = PALETTE[idx % len(PALETTE)]
#                 folium.PolyLine(
#                     [G.nodes[n]["pos"] for n in nodes],
#                     color=colour,
#                     weight=8,
#                     opacity=0.9,
#                     tooltip=f"Path {idx+1}",
#                 ).add_to(m)

#             else:
#                 colour = PALETTE[idx % len(PALETTE)]
#                 folium.PolyLine(
#                     [G.nodes[n]["pos"] for n in nodes],
#                     color=colour,
#                     weight=4,
#                     opacity=0.6,
#                     tooltip=f"Path {idx+1}",
#                 ).add_to(m)

#         lats = [lat for lat,lon in all_coords]
#         lons = [lon for lat,lon in all_coords]
#         m.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])

#         # 3) render the textual list, using the same colours
#         st.subheader("Paths")
#         for idx, (nodes, total_time, total_dist) in enumerate(paths, start=1):
#             colour = PALETTE[(idx-1) % len(PALETTE)]
#             path_str = " → ".join(nodes)
#             if idx == highlight_idx +1:
#                 st.markdown(
#                     f"<span style='color:{colour}'><b><u>"
#                     f"**{idx}.** {total_time:.1f} min, {total_dist:.2f} km — {path_str}"
#                     f"</u></b></span>",
#                     unsafe_allow_html=True
#                 )
#             else:
#                 st.markdown(
#                     f"<span style='color:{colour}'>"
#                     f"**{idx}.** {total_time:.1f} min, {total_dist:.2f} km — {path_str}"
#                     f"</span>",
#                     unsafe_allow_html=True
#                 )

        
#         # 2) put the map on screen
#         st_folium(m, width=800, height=500)
# import streamlit as st
# import pandas as pd
# import folium
# from streamlit_folium import st_folium

# # for volume prediction
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import MinMaxScaler

# from core import load_timeseries, build_weighted_graph, k_shortest_paths, MODEL_MAP
# from models.predictor import LSTMPredictor, GRUPredictor, MLPPredictor, TCNPredictor
# from models.lstm_model import LSTMModel
# from models.gru_model import GRUModel
# from models.mlp_model import MLPModel
# from models.tcn_model import TCNModel

# st.set_page_config(page_title="Part B Path & Volume Explorer", layout="wide")
# st.sidebar.title("Part B Path & Volume Explorer")

# # Initialize session state
# for key in ("G", "paths", "params"): 
#     if key not in st.session_state:
#         st.session_state[key] = None

# # Load centroid nodes
# nodes_df = pd.read_csv("data/scats_complete_average.csv")
# nodes_df["Site_ID"] = nodes_df["Site_ID"].astype(str)
# nodes_df["label"]   = nodes_df["Site_ID"] + " ⎯ " + nodes_df["Location"]
# nodes_df["Location"] = nodes_df["Location"].str.replace("/", " / ", regex=False)

# # --- Sidebar: Path Explorer Inputs ---
# start_sel = st.sidebar.selectbox("Start node", nodes_df["label"], index=0)
# end_sel   = st.sidebar.selectbox("End node",   nodes_df["label"], index=1)

# # Date selectors for October 2006
# with st.sidebar:
#     st.markdown("### Date (October 2006)")
#     c1, c2, c3 = st.columns(3)
#     day    = c1.selectbox("Day",    list(range(1,32)), index=4, key="day")
#     hour   = c2.selectbox("Hour",   list(range(0,24)), index=5, key="hour")
#     minute = c3.selectbox("Minute", [0,15,30,45],     index=1, key="minute")

# try:
#     timestamp = pd.Timestamp(2006, 10, day, hour, minute)
# except ValueError as e:
#     st.sidebar.error(f"Invalid date/time: {e}")
#     st.stop()

# model_name = st.sidebar.selectbox("Prediction model", list(MODEL_MAP.keys()), index=0)
# n_paths    = st.sidebar.number_input("Number of paths", 1, 10, 3)

# if st.sidebar.button("Show paths"):
#     params = {"start": start_sel, "end": end_sel, "ts": timestamp, "model": model_name, "k": n_paths}
#     if params != st.session_state.params:
#         st.session_state.params = params.copy()
#         with st.spinner("Building graph & predicting…"):
#             ts_df = load_timeseries()
#             source = start_sel.split("⎯")[0].strip()
#             target = end_sel.split("⎯")[0].strip()
#             G = build_weighted_graph(ts_df, model_name, timestamp)
#             paths = k_shortest_paths(G, source, target, n_paths)
#         st.session_state.G     = G
#         st.session_state.paths = paths

# # Color palette for routes
# PALETTE = ["red","blue","green","purple","darkred","cadetblue","darkgreen","darkblue","brown"]

# # Render Path Explorer
# if st.session_state.paths:
#     G     = st.session_state.G
#     paths = st.session_state.paths
#     if not paths:
#         st.error("No paths found.")
#     else:
#         # Highlight control and node display options
#         choices = [f"{i}. {t:.1f}m, {d:.2f}km → {' → '.join(ns)}" for i,(ns,t,d) in enumerate(paths,1)]
#         if 'highlight' not in st.session_state:
#             st.session_state.highlight = choices[0]
#         default_idx = 2 if len(choices)>1 else 1
#         if len(choices)==1:
#             disp_opt = st.sidebar.radio("Nodes to display:", ["All nodes","Highlighted path nodes"], index=default_idx-1)
#         else:
#             disp_opt = st.sidebar.radio("Nodes to display:", ["All nodes","All path nodes","Highlighted path nodes"], index=default_idx)
#             st.sidebar.radio("Bring path to front:", choices, key="highlight")
#         hi = int(st.session_state.highlight.split('.')[0]) - 1

#         # Determine nodes to show
#         if disp_opt == "All nodes":
#             show_nodes = list(G.graph['centroids'].keys())
#         elif disp_opt == "All path nodes":
#             show_nodes = sorted({n for (ns,_,_) in paths for n in ns})
#         else:
#             show_nodes = paths[hi][0]

#         # Base map
#         lat0, lon0 = G.nodes[start_sel.split("⎯")[0].strip()]['pos']
#         m = folium.Map(location=[lat0, lon0], zoom_start=13)
#         # Node markers
#         for nid in show_nodes:
#             lat, lon = G.nodes[nid]['pos']
#             loc = nodes_df.loc[nodes_df.Site_ID == nid, 'Location'].iat[0]
#             folium.Marker((lat, lon), popup=f"Node {nid}, {loc}", icon=folium.Icon(color='gray')).add_to(m)
#         # Draw routes
#         all_coords = []
#         for idx,(ns,t,d) in enumerate(paths):
#             coords = [G.nodes[n]['pos'] for n in ns]
#             all_coords.extend(coords)
#             clr = PALETTE[idx % len(PALETTE)]
#             w   = 8 if idx == hi else 4
#             o   = 0.9 if idx == hi else 0.6
#             folium.PolyLine(coords, color=clr, weight=w, opacity=o).add_to(m)
#         # Fit to bounds
#         if all_coords:
#             lats = [lat for lat,lon in all_coords]
#             lons = [lon for lat,lon in all_coords]
#             m.fit_bounds([[min(lats),min(lons)],[max(lats),max(lons)]])
#         st.subheader("Paths")
#         for i,(ns,t,d) in enumerate(paths,1):
#             clr = PALETTE[(i-1)%len(PALETTE)]
#             bold_start = "<b>" if i-1==hi else ""
#             bold_end   = "</b>" if i-1==hi else ""
#             st.markdown(
#                 f"<span style='color:{clr}'>{bold_start}{i}. {t:.1f}m, {d:.2f}km → {' → '.join(ns)}{bold_end}</span>",
#                 unsafe_allow_html=True
#             )
#         st_folium(m, width=800, height=500)

# # --- Traffic Volume Prediction ---
# st.sidebar.markdown("---")
# st.sidebar.header("Traffic Volume Prediction")
# # Reuse site, time, and model inputs
# default_site = start_sel.split("⎯")[0].strip()
# st.sidebar.write(f"Site: **{default_site}**")
# st.sidebar.write(f"Date/Time: **{timestamp:%Y-%m-%d %H:%M}**")
# st.sidebar.write(f"Model: **{model_name}**")
# # Option to use current model or all models
# model_mode = st.sidebar.radio(
#     "Model selection:",
#     ["Current model", "All models"],
#     index=0
# )
# # Location choices filtered by site
# ready_df = pd.read_pickle("data/traffic_model_ready.pkl")
# arms = sorted(ready_df[ready_df.Site_ID.astype(str)==default_site].Location.unique())
# loc_pred = st.sidebar.selectbox("Location", arms)
# # Initialize prediction session state
# if "pred_params" not in st.session_state:
#     st.session_state.pred_params = None
# if "pred_results" not in st.session_state:
#     st.session_state.pred_results = None

# # Run prediction when button clicked
# if st.sidebar.button("Run Prediction", key="run_pred"):
#     pred_params = {
#         "site": default_site,
#         "loc": loc_pred,
#         "day": day,
#         "model_mode": model_mode
#     }
#     if pred_params != st.session_state.pred_params:
#         st.session_state.pred_params = pred_params.copy()
#         with st.spinner("Running volume prediction…"):
#             # Prepare data slice
#             dfp = ready_df.query("Site_ID == @default_site and Location == @loc_pred").sort_values("Timestamp")
#             ts = dfp.Volume.values
#             times = dfp.Timestamp.values
#             SEQ_LEN = 96
#             X_list, y_list = [], []
#             for i in range(SEQ_LEN, len(ts)):
#                 X_list.append(ts[i-SEQ_LEN:i])
#                 y_list.append(ts[i])
#             X_arr = np.stack(X_list).astype(np.float32)
#             y_arr = np.array(y_list).reshape(-1,1).astype(np.float32)
#             times_full = times[SEQ_LEN:]
#             scaler = MinMaxScaler()
#             X_scaled = scaler.fit_transform(X_arr.reshape(-1,1)).reshape(-1,SEQ_LEN)
#             y_scaled = scaler.transform(y_arr)
#             X_tensor = torch.from_numpy(X_scaled).unsqueeze(-1).float()
#             # Build model_map
#             model_map = {
#               "LSTM":(LSTMPredictor, LSTMModel,  f"lstm_saved_models/{default_site}__{loc_pred.replace(' ','_')}.pth"),
#               "GRU": (GRUPredictor,  GRUModel,   f"gru_saved_models/{default_site}__{loc_pred.replace(' ','_')}_GRU.pth"),
#               "MLP": (MLPPredictor,  MLPModel,   f"mlp_saved_models/{default_site}__{loc_pred.replace(' ','_')}_MLP.pth"),
#               "TCN": (TCNPredictor,  TCNModel,   f"tcn_saved_models/{default_site}__{loc_pred.replace(' ','_')}_TCN.pth")
#             }
#             # Determine models to run
#             models_to_run = [model_name] if model_mode == "Current model" else list(model_map.keys())
#             dev = torch.device('cpu')
#             results = {"times_full": times_full, "scaler": scaler, "y_scaled": y_scaled, "models": {}}
#             # Run predictions per model
#             for mname in models_to_run:
#                 PredCls, NetCls, cp = model_map[mname]
#                 ckpt = torch.load(cp, map_location=dev, weights_only=False)
#                 if mname == "MLP":
#                     net = NetCls(input_size=SEQ_LEN, hidden_size=128).to(dev)
#                 elif mname in ("LSTM","GRU"):
#                     net = NetCls(input_size=1, hidden_size=64, num_layers=2).to(dev)
#                 else:
#                     net = NetCls(input_size=1, hidden_size=64, seq_len=SEQ_LEN, output_size=1).to(dev)
#                 net.load_state_dict(ckpt['state_dict'])
#                 net.eval()
#                 with torch.no_grad():
#                     if mname == "MLP":
#                         pr = net(X_tensor.view(X_tensor.size(0), -1).to(dev))
#                     else:
#                         pr = net(X_tensor.to(dev))
#                 pr_np = pr.cpu().numpy()
#                 results["models"][mname] = pr_np
#             # Store results
#             st.session_state.pred_results = results

# # Display prediction plot if available
# if st.session_state.pred_results:
#     res = st.session_state.pred_results
#     params = st.session_state.pred_params
#     times_full = res["times_full"]
#     scaler = res["scaler"]
#     y_scaled = res["y_scaled"]
#     day_start = pd.Timestamp(2006,10,params['day'],0,0)
#     day_end = day_start + pd.Timedelta(days=1)
#     mask = (times_full >= day_start) & (times_full < day_end)
#     times_sel = times_full[mask]
#     actual_sel = scaler.inverse_transform(y_scaled)[mask]
#     fig, ax = plt.subplots(figsize=(10,5))
#     ax.plot(times_sel, actual_sel, label='Actual', color='black')
#     color_cycle = ['blue','orange','green','red']
#     for idx, (mname, pr_np) in enumerate(res['models'].items()):
#         preds_inv = scaler.inverse_transform(pr_np)[mask]
#         ax.plot(
#             times_sel,
#             preds_inv,
#             label=f"{mname} pred",
#             linestyle='--',
#             color=color_cycle[idx % len(color_cycle)]
#         )
#     ax.set_xlim(day_start, day_end)
#     ax.set_xlabel('Time')
#     ax.set_ylabel('Volume')
#     ax.set_title(f'Site {params['site']} — {params['loc']} (2006-10-{params['day']:02d})')
#     ax.legend()
#     ax.grid(True)
#     plt.xticks(rotation=45)
#     st.pyplot(fig, use_container_width=True)

import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium

# for volume prediction
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from core import load_timeseries, build_weighted_graph, k_shortest_paths, MODEL_MAP
from models.predictor import LSTMPredictor, GRUPredictor, MLPPredictor, TCNPredictor
from models.lstm_model import LSTMModel
from models.gru_model import GRUModel
from models.mlp_model import MLPModel
from models.tcn_model import TCNModel

st.set_page_config(page_title="Part B Path & Volume Explorer", layout="wide")
st.sidebar.title("Part B Controls")

# Shared date/time inputs (October 2006)
st.sidebar.markdown("### Date (October 2006)")
c1, c2, c3 = st.sidebar.columns(3)
day    = c1.selectbox("Day",    list(range(1,32)), index=4, key="day")
hour   = c2.selectbox("Hour",   list(range(0,24)), index=5, key="hour")
minute = c3.selectbox("Minute", [0,15,30,45],     index=1, key="minute")
try:
    timestamp = pd.Timestamp(2006, 10, day, hour, minute)
except ValueError as e:
    st.sidebar.error(f"Invalid date/time: {e}")
    st.stop()

# Sidebar inputs for both features
def load_ready():
    return pd.read_pickle("data/traffic_model_ready.pkl")
ready_df = load_ready()

# Load nodes for path explorer dropdowns
nodes_df = pd.read_csv("data/scats_complete_average.csv")
nodes_df["Site_ID"] = nodes_df["Site_ID"].astype(str)
nodes_df["label"]   = nodes_df["Site_ID"] + " ⎯ " + nodes_df["Location"]
nodes_df["Location"] = nodes_df["Location"].str.replace("/", " / ", regex=False)

# Tabs for features
tab_paths, tab_vol = st.tabs(["Path Explorer", "Volume Prediction"])

# Path Explorer Feature
with tab_paths:
    st.header("Path Explorer")
    # Sidebar controls
    start_sel   = st.sidebar.selectbox("Start node", nodes_df["label"], key="p_start")
    end_sel     = st.sidebar.selectbox("End node",   nodes_df["label"], key="p_end")
    model_name  = st.sidebar.selectbox("Prediction model", list(MODEL_MAP.keys()), key="p_model")
    n_paths     = st.sidebar.number_input("Number of paths", 1, 10, 3, key="p_k")
    if st.sidebar.button("Show paths", key="show_paths"):
        # compute graph & paths
        with st.spinner("Building graph & predicting…"):
            ts_df = load_timeseries()
            source = start_sel.split("⎯")[0].strip()
            target = end_sel.split("⎯")[0].strip()
            G = build_weighted_graph(ts_df, model_name, timestamp)
            paths = k_shortest_paths(G, source, target, n_paths)
            st.session_state.paths = paths
            st.session_state.G = G
    # display results
    paths = st.session_state.get("paths")
    if paths:
        G = st.session_state.G
        PALETTE = ["red","blue","green","purple","darkred","cadetblue","darkgreen","darkblue","brown"]
        # render map
        all_coords = []
        m = folium.Map(location=[0,0], zoom_start=13)
        for idx,(nodes, t, d) in enumerate(paths):
            coords = [G.nodes[n]["pos"] for n in nodes]
            all_coords.extend(coords)
            folium.PolyLine(coords, color=PALETTE[idx], weight=4, tooltip=f"Path {idx+1}").add_to(m)
        if all_coords:
            lats = [lat for lat,lon in all_coords]; lons=[lon for lat,lon in all_coords]
            m.fit_bounds([[min(lats),min(lons)],[max(lats),max(lons)]])
        st_folium(m, width=800, height=500)
        # list
        st.subheader("Paths")
        for i,(nodes,t,d) in enumerate(paths,1):
            st.markdown(f"**{i}.** {t:.1f} min, {d:.2f} km → {' → '.join(nodes)}")
    else:
        st.info("Click **Show paths** in the sidebar to compute routes.")

# Volume Prediction Feature
pred_params = st.session_state.get("pred_params", {})
if not pred_params:
    pred_params = {
        "site": None,
        "loc": None,
        "day": day,
        "model_mode": "Current model"
    }

with tab_vol:
    st.header("Traffic Volume Prediction")
    # reuse date/time, but site from path explorer start_sel
    site = start_sel.split("⎯")[0].strip() if 'p_start' in st.session_state else None
    # st.subheader(f"Site: {site or '—'}  |  Time: {timestamp:%Y-%m-%d %H:%M}")
    # location dropdown for that site
    arms = sorted(ready_df[ready_df.Site_ID.astype(str)==site].Location.unique()) if site else []
    loc = st.sidebar.selectbox("Location", arms, key="vol_loc")
    mode = st.sidebar.radio("Models to run", ["Current model","All models"], key="vol_mode")
    if st.sidebar.button("Run Prediction", key="vol_run"):
        pred_params = {
            "site": site,
            "loc": loc,
            "day": day,
            "model_mode": mode
        }
        if pred_params != st.session_state.get("pred_params"):
            st.session_state.pred_params = pred_params.copy()
            # run prediction

        with st.spinner("Running volume prediction…"):
            # prepare data
            dfp = ready_df.query("Site_ID==@site and Location==@loc").sort_values("Timestamp")
            ts = dfp.Volume.values
            times = dfp.Timestamp.values
            SEQ_LEN = 96
            Xl, yl = [], []
            for i in range(SEQ_LEN,len(ts)):
                Xl.append(ts[i-SEQ_LEN:i]); yl.append(ts[i])
            X_arr = np.stack(Xl).astype(np.float32)
            y_arr = np.array(yl).reshape(-1,1).astype(np.float32)
            times_full = times[SEQ_LEN:]
            scaler = MinMaxScaler()
            Xs = scaler.fit_transform(X_arr.reshape(-1,1)).reshape(-1,SEQ_LEN)
            ys = scaler.transform(y_arr)
            Xt = torch.from_numpy(Xs).unsqueeze(-1).float()
            # model definitions
            model_map = {
              "LSTM":(LSTMPredictor,LSTMModel, f"lstm_saved_models/{site}__{loc.replace(' ','_')}.pth"),
              "GRU": (GRUPredictor,GRUModel,  f"gru_saved_models/{site}__{loc.replace(' ','_')}_GRU.pth"),
              "MLP": (MLPPredictor,MLPModel,  f"mlp_saved_models/{site}__{loc.replace(' ','_')}_MLP.pth"),
              "TCN": (TCNPredictor,TCNModel,  f"tcn_saved_models/{site}__{loc.replace(' ','_')}_TCN.pth")
            }
            to_run = [model_name] if mode=="Current model" else list(model_map.keys())
            dev = torch.device('cpu')
            preds_dict = {}
            for m in to_run:
                _,NetCls,cp = model_map[m]
                ck = torch.load(cp, map_location=dev, weights_only=False)
                if m=="MLP": net = NetCls(input_size=SEQ_LEN,hidden_size=128).to(dev)
                elif m in ("LSTM","GRU"): net = NetCls(input_size=1,hidden_size=64,num_layers=2).to(dev)
                else: net = NetCls(input_size=1,hidden_size=64,seq_len=SEQ_LEN,output_size=1).to(dev)
                net.load_state_dict(ck['state_dict']); net.eval()
                with torch.no_grad():
                    out = net(Xt.view(Xt.size(0),-1).to(dev)) if m=="MLP" else net(Xt.to(dev))
                preds_dict[m] = out.cpu().numpy()
            st.session_state.vol_results = {
                'times': times_full,
                'scaler': scaler,
                'ys': ys,
                'preds': preds_dict
            }
    # display results
    if 'vol_results' in st.session_state:
        res = st.session_state.vol_results
        params = st.session_state.get("pred_params", {})
        times_full = res["times"]
        scaler = res["scaler"]
        y_scaled = res["ys"]
        day_start = pd.Timestamp(2006,10,params['day'],0,0)
        day_end = day_start + pd.Timedelta(days=1)
        mask = (times_full >= day_start) & (times_full < day_end)
        times_sel = times_full[mask]
        actual_sel = scaler.inverse_transform(y_scaled)[mask]
        fig, ax = plt.subplots(figsize=(12,6))
        ax.plot(times_sel, actual_sel, label='Actual', color='black')
        color_cycle = ['blue','orange','green','red']
        for idx,(mname,pr_np) in enumerate(res['preds'].items()):
            preds_inv = scaler.inverse_transform(pr_np)[mask]
            ax.plot(times_sel, preds_inv, linestyle='--', color=color_cycle[idx%len(color_cycle)], label=f"{mname} pred")
        ax.set_xlim(day_start, day_end)
        ax.set_xlabel('Time'); ax.set_ylabel('Volume')
        ax.set_title(f'Site {params['site']} — {params['loc']} (2006-10-{params['day']:02d})')
        ax.legend(); ax.grid(True)
        plt.xticks(rotation=45)
        st.pyplot(fig, use_container_width=True)
    else:
        st.info("Click 'Run Prediction' in the sidebar to generate volume forecasts.")

