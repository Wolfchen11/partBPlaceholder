
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

def load_ready():
    return pd.read_pickle("data/traffic_model_ready.pkl")
ready_df = load_ready()

# Load nodes for path explorer dropdowns
nodes_df = pd.read_csv("data/scats_complete_average.csv")
nodes_df["Site_ID"] = nodes_df["Site_ID"].astype(str)
nodes_df["label"]   = nodes_df["Site_ID"] + " ⎯ " + nodes_df["Location"]
nodes_df["Location"] = nodes_df["Location"].str.replace("/", " / ", regex=False)

st.set_page_config(page_title="Part B Path & Volume Explorer", layout="wide")
st.sidebar.title("Part B Controls")
start_sel   = st.sidebar.selectbox("Start node", nodes_df["label"], key="p_start", index=0)
end_sel     = st.sidebar.selectbox("End node",   nodes_df["label"], key="p_end", index=1)

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

model_name  = st.sidebar.selectbox("Prediction model", list(MODEL_MAP.keys()), key="p_model")
n_paths     = st.sidebar.number_input("Number of paths", 1, 10, 3, key="p_k")

# Tabs for features
tab_paths, tab_vol = st.tabs(["Path Explorer", "Volume Prediction"])

# Path Explorer Feature
with tab_paths:
    # Sidebar controls
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
        PALETTE = ["red","blue","green","purple","brown","darkred","cadetblue","darkgreen","darkblue"]
        choices = [f"{i}. {t:.1f}m, {d:.2f}km → {' → '.join(ns)}" for i,(ns,t,d) in enumerate(paths,1)]
        if 'highlight' not in st.session_state:
            st.session_state.highlight = choices[0]
        default_idx = 2 if len(choices) > 1 else 1

        st.markdown("##### Display Controls")
        # st.subheader("Display Controls")

        if len(choices) == 1:
            disp_opt = st.radio(
                "Nodes to display:",
                ["All nodes", "Highlighted path nodes"],
                index=default_idx-1,
                key="disp_opt"
            )
        else:
            disp_opt = st.radio(
                "Nodes to display:",
                ["All nodes", "All path nodes", "Highlighted path nodes"],
                index=default_idx,
                key="disp_opt"
            )

        hi = st.session_state.get("highlight_idx", 0)

        # Now determine which nodes to show
        if disp_opt == "All nodes":
            show_nodes = list(G.graph['centroids'].keys())
        elif disp_opt == "All path nodes":
            show_nodes = sorted({n for (ns,_,_) in paths for n in ns})
        else:
            show_nodes = paths[hi][0]

        st.markdown("##### Select and highlight a path")
        for idx, (nodes, t, d) in enumerate(paths):
        # build label & styling
            color = PALETTE[idx % len(PALETTE)]
            label = f"{idx+1}. {t:.1f}m, {d:.2f}km → {' → '.join(nodes)}"
            if idx == hi:
                styled = (
                    f"<span style='color:{color}; "
                    f"font-weight:bold; text-decoration:underline'>{label}</span>"
                )
            else:
                styled = f"<span style='color:{color}'>{label}</span>"

            # make two columns: 1 for the button, 9 for the text
            col_btn, col_txt = st.columns([1, 9])
            with col_btn:
                if st.button("Select", key=f"btn_{idx}"):
                    st.session_state.highlight_idx = idx
                    hi = idx
            with col_txt:
                st.markdown(styled, unsafe_allow_html=True)

        # Base map
        lat0, lon0 = G.nodes[start_sel.split("⎯")[0].strip()]['pos']
        m = folium.Map(location=[lat0, lon0], zoom_start=13)
        # Node markers
        for nid in show_nodes:
            lat, lon = G.nodes[nid]['pos']
            loc = nodes_df.loc[nodes_df.Site_ID == nid, 'Location'].iat[0]
            folium.Marker((lat, lon), popup=f"Node {nid}, {loc}", icon=folium.Icon(color='gray')).add_to(m)
        # Draw routes
        all_coords = []
        for idx,(ns,t,d) in enumerate(paths):
            coords = [G.nodes[n]['pos'] for n in ns]
            all_coords.extend(coords)
            clr = PALETTE[idx % len(PALETTE)]
            w   = 8 if idx == hi else 4
            o   = 0.9 if idx == hi else 0.6
            folium.PolyLine(coords, color=clr, weight=w, opacity=o).add_to(m)
        # Fit to bounds
        if all_coords:
            lats = [lat for lat,lon in all_coords]
            lons = [lon for lat,lon in all_coords]
            m.fit_bounds([[min(lats),min(lons)],[max(lats),max(lons)]])

        st_folium(m, width=800, height=500)
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

