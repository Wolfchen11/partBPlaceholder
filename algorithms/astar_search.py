# # algorithms/astar_search.py

# from collections import defaultdict
# import heapq
# import pandas as pd

# from utils.geo_utils import haversine
# from utils.flow_to_speed import flow_to_speed
# from utils.edge_mapper import EdgeMapper

# # Initialize the mapper once
# mapper = EdgeMapper(
#     arms_pkl="data/traffic_model_ready.pkl",
#     nodes_csv="data/scats_complete.csv"
# )

# def astar(start, goal, centroids, edges, predictor, start_timestamp, k=3):
#     """
#     Find up to k best routes (by estimated travel time, in minutes)
#     from `start` to `goal`, given:
#       - centroids: dict node → (lat, lon)
#       - edges: list of (A, B, dist_km)
#       - predictor: UnifiedPredictor instance
#       - start_timestamp: string or pd.Timestamp
#       - k: how many top routes to return

#     Returns a list of (path_list, total_time_minutes) tuples.
#     """
#     start = str(start)
#     goal  = str(goal)

#     # 1) Build adjacency list
#     adjacency = defaultdict(list)
#     for A, B, dist in edges:
#         adjacency[A].append((B, dist))

#     # 2) Cache for predictor: (node, arm, departure_time) → flow
#     flow_cache = {}

#     # 3) Priority queue: (f = g + h, g, current, dep_time, path)
#     frontier = []
#     start_dep = pd.to_datetime(start_timestamp)
#     # Heuristic at start: straight‐line (km) → optimistic minutes at 60 km/h
#     h0 = haversine(*centroids[start], *centroids[goal])
#     heapq.heappush(frontier, (h0, 0.0, start, start_dep, [start]))

#     found = []

#     # 4) Main loop
#     while frontier and len(found) < k:
#         f, g, current, dep_time, path = heapq.heappop(frontier)

#         # If we reached goal, record and continue
#         if current == goal:
#             found.append((path, g))
#             print(f"Found path: {path} with total time {g:.1f} min")
#             continue

#         # Expand neighbors
#         for B, dist_km in adjacency[current]:
#             # Determine which arm (SCATS site/arm) to use
#             loc = mapper.best_arm(current, B, centroids)
#             key = (current, loc, dep_time)

#             # Fetch or compute predicted flow
#             if key in flow_cache:
#                 flow = flow_cache[key]
#             else:
#                 # print(f"Predicting flow for {current}, {loc}, {dep_time}")
#                 flow = predictor.predict(current, loc, dep_time)
#                 flow_cache[key] = flow

#             # Convert flow → speed → travel time (minutes)
#             speed = flow_to_speed(flow)
#             travel_min = dist_km / speed * 60 + 0.5

#             new_g   = g + travel_min
#             new_dep = dep_time + pd.Timedelta(minutes=travel_min)
#             # Heuristic from B → goal
#             h       = haversine(*centroids[B], *centroids[goal])
#             priority = new_g + h

#             # Push new state
#             heapq.heappush(
#                 frontier,
#                 (priority, new_g, B, new_dep, path + [B])
#             )

#     for i in range(len(found)):
#         path, total_time = found[i]
#         # Convert to km
#         total_dist = 0.0
#         for u, v in zip(path, path[1:]):
#             for A, B, dist in edges:
#                 if u == A and v == B:
#                     total_dist += dist
#                     break
#         found[i] = (path, total_time, total_dist)    

#     return found

from collections import defaultdict
import heapq
import pandas as pd

from utils.geo_utils import haversine
from utils.flow_to_speed import flow_to_speed
from utils.edge_mapper import EdgeMapper

# Initialize the mapper once
mapper = EdgeMapper(
    arms_pkl="data/traffic_model_ready.pkl",
    nodes_csv="data/scats_complete.csv"
)

# A* search that avoids cycles and skips missing models

def astar(start, goal, centroids, edges, predictor, start_timestamp, k=3):
    """
    Find up to k best routes by estimated travel time (minutes)
    using A* search. Returns list of (path, total_time, total_dist).
    """
    start = str(start)
    goal  = str(goal)

    # Build adjacency list
    adjacency = defaultdict(list)
    for A, B, dist in edges:
        adjacency[A].append((B, dist))

    # Cache for predictor: (node, arm, departure_time) → flow
    flow_cache = {}

    # Priority queue: (f = g + h, g, current, dep_time, path)
    frontier = []
    start_dep = pd.to_datetime(start_timestamp)
    # Heuristic at start: straight-line (km) → optimistic minutes at 60 km/h
    h0 = haversine(*centroids[start], *centroids[goal])
    heapq.heappush(frontier, (h0, 0.0, start, start_dep, [start]))

    found = []
    # Main loop
    while frontier and len(found) < k:
        f, g, current, dep_time, path = heapq.heappop(frontier)

        # If reached goal, record path and cost
        if current == goal:
            found.append((path, g))
            continue

        # Expand neighbors, skipping already visited nodes to avoid cycles
        for B, dist_km in adjacency[current]:
            if B in path:
                continue

            # Select the correct arm
            loc = mapper.best_arm(current, B, centroids)
            key = (current, loc, dep_time)

            # Fetch or predict flow, skipping if model missing
            if key in flow_cache:
                flow = flow_cache[key]
            else:
                try:
                    flow = predictor.predict(current, loc, dep_time)
                except FileNotFoundError:
                    continue
                except Exception:
                    continue
                flow_cache[key] = flow

            # Convert flow → speed → travel time
            if not flow or flow <= 0:
                continue
            speed = flow_to_speed(flow)
            travel_min = dist_km / speed * 60 + 0.5

            new_g   = g + travel_min
            new_dep = dep_time + pd.Timedelta(minutes=travel_min)
            # Heuristic from B to goal
            h = haversine(*centroids[B], *centroids[goal])

            heapq.heappush(
                frontier,
                (new_g + h, new_g, B, new_dep, path + [B])
            )

    # Compile results with distance
    results = []
    for path, total_time in found:
        total_dist = 0.0
        for u, v in zip(path, path[1:]):
            for A, B, dist in edges:
                if u == A and v == B:
                    total_dist += dist
                    break
        results.append((path, total_time, total_dist))
    return results
