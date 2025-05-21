#!/usr/bin/env python3
# main.py

import argparse
from algorithms.graph_builder import build_graph
from algorithms.astar_search   import astar
from utils.edge_mapper         import EdgeMapper
from models.predictor import LSTMPredictor, GRUPredictor, MLPPredictor, TCNPredictor
from utils.flow_to_speed       import flow_to_speed


def main():
    p = argparse.ArgumentParser(
        description="TBRGS: Traffic‑Based Route Guidance System"
    )
    p.add_argument('--source',    required=True, help='Origin site ID (e.g. 0970)')
    p.add_argument('--target',    required=True, help='Destination site ID (e.g. 3685)')
    p.add_argument('--timestamp', required=True,
                   help='Timestamp for prediction (YYYY-MM-DD HH:MM:SS)')
    p.add_argument('--model', required=True, help = 'Model to use (LSTM or GRU)')
    p.add_argument('--nodes',     default='data/scats_complete_average.csv',
                   help='Path to node centroids CSV')
    p.add_argument('--volumes',   default='data/traffic_model_ready.pkl',
                   help='Path to volume pickle')
    args = p.parse_args()

    # 1) Build graph
    print(f"🔍 Building graph from {args.nodes} …")
    centroids, edges = build_graph(args.nodes)

    # 2) Instantiate mapper & predictor
    print("🗺️  Initializing edge→arm mapper & LSTM predictor …")
    mapper    = EdgeMapper(args.volumes)
    if args.model.upper() == 'LSTM':
        predictor = LSTMPredictor(data_pkl=args.volumes,
                                  models_dir="lstm_saved_models")
    elif args.model.upper() == 'GRU':
        predictor = GRUPredictor(data_pkl=args.volumes,
                                 models_dir="gru_saved_models")
    elif args.model.upper() == 'MLP':
        predictor = MLPPredictor(data_pkl=args.volumes,
                                 models_dir="mlp_saved_models")
    elif args.model.upper() == 'TCN':
        predictor = TCNPredictor(data_pkl=args.volumes,
                                 models_dir="tcn_saved_models")

    # 4) Run A* to get the fastest route under predicted traffic
    print(f"🚦 Running A* from {args.source} → {args.target} at {args.timestamp} …")
    #path, total_time = astar(args.source, args.target,centroids, edges, predictor, args.timestamp)
    paths = astar(args.source, args.target,centroids, edges, predictor, args.timestamp, k=3)

    if not paths:
        print("❌ No route found.")
        return

    # # 5) Compute total distance
    # total_dist = 0.0
    # # build a quick lookup of (u→v) distances
    # dist_map = {(u, v): d for u, v, d in edges}
    # for u, v in zip(path, path[1:]):
    #     total_dist += dist_map.get((u, v), 0.0)

    # 6) Print results
    i = 0
    for path in paths:
        i+= 1
        print(f"\n🛣️ Optimal route {i}:")
        print("   " + " → ".join(path[0]))
        print(f"\n📏 Total distance: {path[2]:.2f} km")
        print(f"⏱️ Total travel time: {path[1]:.1f} minutes")

if __name__ == "__main__":
    main()
