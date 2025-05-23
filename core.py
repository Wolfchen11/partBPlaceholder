import pandas as pd
import networkx as nx
import os

from utils.data_loader import load_node_data
from utils.flow_to_speed import flow_to_speed
from models.predictor import LSTMPredictor, GRUPredictor, MLPPredictor, TCNPredictor
from utils.geo_utils import haversine
from algorithms.graph_builder import build_graph
from algorithms.astar_search import astar

# Map friendly names to predictor classes and their model directories
MODEL_MAP = {
    "LSTM": (LSTMPredictor, "lstm_saved_models", "lstm_saved_models/"),
    "GRU":  (GRUPredictor,  "gru_saved_models", "gru_saved_models/"),
    "MLP":  (MLPPredictor,  "mlp_saved_models", "mlp_saved_models/"),
    "TCN":  (TCNPredictor,  "tcn_saved_models", "tcn_saved_models/"),
}

def load_timeseries(ts_csv: str = "data/traffic_model_ready.csv") -> pd.DataFrame:
    df = pd.read_csv(ts_csv)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    return df


def build_weighted_graph(
    ts_df: pd.DataFrame,
    model_name: str,
    timestamp: pd.Timestamp,
    node_csv: str = "data/scats_complete_average.csv",
) -> nx.DiGraph:
    """
    Builds a directed graph with edge weights based on predicted travel-time.

    - Uses `build_graph` to get centroids & edges.
    - Instantiates the chosen predictor and queries per site.
    - Attaches centroids, edges, predictor, and timestamp onto G.graph for A*.
    """
    # locate the pickled time-series for prediction
    data_pkl = os.path.join(os.path.dirname(__file__), "data", "traffic_model_ready.pkl")

    # 1) static topology + coords
    centroids, edges = build_graph(node_csv)

    # 2) predictor instantiation
    PredictorCls, models_dir1, models_dir2 = MODEL_MAP[model_name]
    pred = PredictorCls(data_pkl, models_dir1, models_dir2)

    # 2a) per-site prediction
    speed_map: dict[str, float] = {}
    for site_id in centroids:
        try:
            speed_map[site_id] = pred.predict(int(site_id), timestamp)
        except Exception:
            speed_map[site_id] = None

    # 3) build graph with travel-time weights (minutes)
    G = nx.DiGraph()
    for sid, (lat, lon) in centroids.items():
        G.add_node(sid, pos=(lat, lon))
    for A, B, dist_km in edges:
        sp = speed_map.get(A)
        wt = (dist_km / sp * 60) if sp and sp > 0 else float('inf')
        G.add_edge(A, B, weight=wt)

    # stash for A*:
    G.graph['centroids'] = centroids
    G.graph['edges']     = edges
    G.graph['predictor']  = pred
    G.graph['timestamp']  = timestamp

    return G


def k_shortest_paths(
    G: nx.DiGraph,
    source: str,
    target: str,
    k: int
) -> list[tuple[list[str], float, float]]:
    """
    Uses the custom A* to return up to k routes.

    Returns:
      List of tuples: (path_nodes, total_time_minutes, total_dist_km)
    """
    centroids = G.graph['centroids']
    edges     = G.graph['edges']
    pred      = G.graph['predictor']
    ts        = G.graph['timestamp']

    results = astar(
        start=source,
        goal=target,
        centroids=centroids,
        edges=edges,
        predictor=pred,
        start_timestamp=ts,
        k=k
    )
    return results
