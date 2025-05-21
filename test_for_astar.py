# test_for_astar.py

from algorithms.graph_builder import build_graph
from algorithms.astar_search import astar
from utils.edge_mapper    import EdgeMapper
from models.predictor import LSTMPredictor, GRUPredictor, MLPPredictor, TCNPredictor
# from utils.flow_to_speed  import flow_to_speed


# # 1) Build the graph
centroids, edges = build_graph("data/scats_complete_average.csv")

# # 2) Instantiate the mapper and predictor
mapper    = EdgeMapper("data/traffic_model_ready.pkl")
# predictor = LSTMPredictor(
#     data_pkl="data/traffic_model_ready.pkl",
#     models_dir="models/lstm_saved_models"
# )
# predictor = GRUPredictor(
#     data_pkl="data/traffic_model_ready.pkl",
#     models_dir="models/gru_saved_models"
# )
# predictor = MLPPredictor(
#     data_pkl="data/traffic_model_ready.pkl",
#     models_dir="models/mlp_saved_models"
# )
predictor = TCNPredictor(
    data_pkl="data/traffic_model_ready.pkl",
    models_dir="models/tcn_saved_models"
)


routes = astar(
    start=2000, #4063
    goal=4321,
    centroids=centroids,
    edges=edges,
    predictor=predictor,
    start_timestamp="2006-10-08 14:45:00",
    k=5 # number of routes to return
)


for i, (path, time_min) in enumerate(routes, 1):
    total_dist = 0.0
    for u, v, d in edges:
        if u in path:
            try:
                idx = path.index(u)
                if path[idx+1] == v:
                    total_dist += d
            except (IndexError, ValueError):
                pass
    print(f"Route #{i}: {path} → {time_min:.1f} min, {total_dist:.2f} km")