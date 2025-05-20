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
predictor = LSTMPredictor(
    data_pkl="data/traffic_model_ready.pkl",
    models_dir="models/lstm_saved_models"
)
# predictor = GRUPredictor(
#     data_pkl="data/traffic_model_ready.pkl",
#     models_dir="models/gru_saved_models"
# )
# predictor = MLPPredictor(
#     data_pkl="data/traffic_model_ready.pkl",
#     models_dir="models/mlp_saved_models"
# )
# predictor = TCNPredictor(
#     data_pkl="data/traffic_model_ready.pkl",
#     models_dir="models/tcn_saved_models"
# )

# # 3) Choose your test parameters
# source    = "4063"                     # origin site
# target    = "4321"                     # destination site
# timestamp = "2006-10-08 14:45:00"      # when to predict

# # 4) Define the real get_volume function
# def get_volume_at_edge(A, B):
#     # pick the right arm at A for edge A->B
#     loc = mapper.best_arm(A, B, centroids)
#     # predict the volume for that arm at the given timestamp
#     return predictor.predict(A, loc, timestamp)

# # 5) Run A* search
# path, total_time = astar(source, target, centroids, edges, predictor, timestamp)

# # 6) (Optional) Compute total distance along that path
# total_dist = 0.0
# for u, v, d in edges:
#     if u in path:
#         try:
#             idx = path.index(u)
#             if path[idx+1] == v:
#                 total_dist += d
#         except (IndexError, ValueError):
#             pass

# # 7) Print results
# print(f"\n🚗  Route from {source} → {target} at {timestamp}:")
# print("    " + " → ".join(path))
# print(f"📏  Total distance: {total_dist:.2f} km")
# print(f"⏱️  Total travel time: {total_time:.1f} minutes")


routes = astar(
    start=4063,
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