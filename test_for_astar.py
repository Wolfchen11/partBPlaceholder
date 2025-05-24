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
    start=970, #4063
    goal=2000,
    centroids=centroids,
    edges=edges,
    predictor=predictor,
    start_timestamp="2006-10-05 05:15:00",
    k=3 # number of routes to return
)


for i, route in enumerate(routes, 1):
    print(f"Route #{i}: {route[0]} → {route[1]:.1f} min, {route[2]:.2f} km")