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
# partBPlaceholder/streamlit_app.py
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
st.sidebar.title("Part B Path & Volume Explorer")

# ──────────────────────────────────────────────────────────────────────────────
# Session state initialization
for key in ("G", "paths", "params"): 
    if key not in st.session_state:
        st.session_state[key] = None

# Load node list for pathing
nodes_df = pd.read_csv("data/scats_complete_average.csv")
nodes_df["Site_ID"] = nodes_df["Site_ID"].astype(str)
nodes_df["label"]   = nodes_df["Site_ID"] + " ⎯ " + nodes_df["Location"]
nodes_df["Location"] = nodes_df["Location"].str.replace("/", " / ", regex=False)

# --- Sidebar: Path Explorer Inputs ---
start_sel = st.sidebar.selectbox("Start node", nodes_df["label"], index=0)
end_sel   = st.sidebar.selectbox("End node",   nodes_df["label"], index=1)

with st.sidebar:
    st.markdown("### Date (October 2006)")
    c1, c2, c3 = st.columns(3)
    day    = c1.selectbox("Day",    list(range(1,32)), index=4, key="day")
    hour   = c2.selectbox("Hour",   list(range(0,24)), index=5, key="hour")
    minute = c3.selectbox("Minute", [0,15,30,45],     index=1, key="minute")

try:
    timestamp = pd.Timestamp(2006,10,day,hour,minute)
except ValueError as e:
    st.sidebar.error(f"Invalid date/time: {e}")
    st.stop()

model_name = st.sidebar.selectbox("Prediction model", list(MODEL_MAP.keys()), index=0)
n_paths    = st.sidebar.number_input("Number of paths", 1, 10, 3)

if st.sidebar.button("Show paths"):
    params = {"start":start_sel, "end":end_sel, "ts":timestamp, "model":model_name, "k":n_paths}
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

PALETTE = ["red","blue","green","purple","darkred","cadetblue","darkgreen","darkblue","brown"]

# --- Render Paths ---
if st.session_state.paths:
    G     = st.session_state.G
    paths = st.session_state.paths
    if not paths:
        st.error("No paths found.")
    else:
        # highlight control
        choices = [f"{i}. {t:.1f}m, {d:.2f}km → {'→'.join(ns)}" for i,(ns,t,d) in enumerate(paths,1)]
        if 'highlight' not in st.session_state:
            st.session_state.highlight = choices[0]
        default_idx = 2 if paths else 0
        if len(choices)==1:
            disp_opt = st.sidebar.radio("Nodes to display:",["All nodes","Highlighted path nodes"],index=default_idx-1)
        else:
            disp_opt = st.sidebar.radio("Nodes to display:",["All nodes","All path nodes","Highlighted path nodes"],index=default_idx)
            st.sidebar.radio("Bring path to front:",choices,key="highlight")
        hi = int(st.session_state.highlight.split('.')[0]) - 1
        if disp_opt=="All nodes":
            show_nodes = list(G.graph['centroids'].keys())
        elif disp_opt=="All path nodes":
            show_nodes = sorted({n for(ns,_,_) in paths for n in ns})
        else:
            show_nodes = paths[hi][0]
        # map
        lat0,lon0 = G.nodes[start_sel.split("⎯")[0].strip()]['pos']
        m = folium.Map(location=[lat0,lon0],zoom_start=13)
        for nid in show_nodes:
            lat,lon = G.nodes[nid]['pos']
            loc = nodes_df.loc[nodes_df.Site_ID==nid,'Location'].iat[0]
            folium.Marker((lat,lon),popup=f"{nid}, {loc}",icon=folium.Icon(color='gray')).add_to(m)
        allc=[]
        for idx,(ns,t,d) in enumerate(paths):
            cs=[G.nodes[n]['pos'] for n in ns]
            allc+=cs
            clr=PALETTE[idx%len(PALETTE)]
            w=8 if idx==hi else 4
            o=0.9 if idx==hi else 0.6
            folium.PolyLine(cs,color=clr,weight=w,opacity=o).add_to(m)
        lats=[lat for lat,lon in allc]; lons=[lon for lat,lon in allc]
        m.fit_bounds([[min(lats),min(lons)],[max(lats),max(lons)]])
        st.subheader("Paths")
        for i,(ns,t,d) in enumerate(paths,1):
            clr=PALETTE[(i-1)%len(PALETTE)]
            bold = "<b>" if i-1==hi else ""
            endb = "</b>" if i-1==hi else ""
            st.markdown(f"<span style='color:{clr}'>{bold}{i}. {t:.1f}m, {d:.2f}km → {'→'.join(ns)}{endb}</span>",unsafe_allow_html=True)
        st_folium(m,width=800,height=500)

# --- Traffic Volume Prediction ---
st.sidebar.markdown("---")
st.sidebar.header("Traffic Volume Prediction")
# reuse path inputs for site/model/time
default_site = start_sel.split("⎯")[0].strip()
st.sidebar.write(f"Site: {default_site}")
st.sidebar.write(f"Date/Time: {timestamp:%Y-%m-%d %H:%M}")
st.sidebar.write(f"Model: {model_name}")
# location dropdown based on site
ready = pd.read_pickle("data/traffic_model_ready.pkl")
arms = sorted(ready[ready.Site_ID.astype(str)==default_site].Location.unique())
loc_pred = st.sidebar.selectbox("Location", arms)
if st.sidebar.button("Run Prediction", key="run_pred"):
    dfp = ready.query("Site_ID == @default_site and Location == @loc_pred").sort_values("Timestamp")
    ts = dfp.Volume.values; times = dfp.Timestamp.values
    SEQ=96; Xl=[]; yl=[]
    for i in range(SEQ,len(ts)): Xl.append(ts[i-SEQ:i]); yl.append(ts[i])
    Xa=np.stack(Xl).astype(np.float32); ya=np.array(yl).reshape(-1,1).astype(np.float32); times=times[SEQ:]
    scaler=MinMaxScaler(); Xs=scaler.fit_transform(Xa.reshape(-1,1)).reshape(-1,SEQ); ys=scaler.transform(ya)
    Xt=torch.from_numpy(Xs).unsqueeze(-1).float(); dev=torch.device('cpu')
    mmap={ "LSTM":(LSTMPredictor,LSTMModel,f"lstm_saved_models/{default_site}__{loc_pred.replace(' ','_')}.pth"),
           "GRU":(GRUPredictor,GRUModel, f"gru_saved_models/{default_site}__{loc_pred.replace(' ','_')}_GRU.pth"),
           "MLP":(MLPPredictor,MLPModel, f"mlp_saved_models/{default_site}__{loc_pred.replace(' ','_')}_MLP.pth"),
           "TCN":(TCNPredictor,TCNModel,f"tcn_saved_models/{default_site}__{loc_pred.replace(' ','_')}_TCN.pth") }
    PredCls,NetCls,cp = mmap[model_name]
    ck=torch.load(cp,map_location=dev, weights_only=False); 
    if model_name=="MLP": net=NetCls(input_size=SEQ,hidden_size=128).to(dev)
    elif model_name in ("LSTM","GRU"): net=NetCls(input_size=1,hidden_size=64,num_layers=2).to(dev)
    else: net=NetCls(input_size=1,hidden_size=64,seq_len=SEQ,output_size=1).to(dev)
    net.load_state_dict(ck['state_dict']); net.eval()
    with torch.no_grad():
        pr = net(Xt.view(Xt.size(0),-1).to(dev)) if model_name=="MLP" else net(Xt.to(dev))
    preds=pr.cpu().numpy(); invp=scaler.inverse_transform(preds); invy=scaler.inverse_transform(ys)
    fig,ax=plt.subplots(figsize=(10,4)); ax.plot(times,invy,label='Actual',color='k'); ax.plot(times,invp,label='Pred',linestyle='--')
    ax.set_xlabel('Time'); ax.set_ylabel('Volume'); ax.set_title(f"Site {default_site} — {loc_pred}"); ax.legend(); ax.grid(True)
    plt.xticks(rotation=45); st.pyplot(fig)
