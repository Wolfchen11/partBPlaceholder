from models.predictor import LSTMPredictor, GRUPredictor, MLPPredictor, TCNPredictor

# LSTMTrainer = LSTMPredictor(
#     data_pkl="data/traffic_model_ready.pkl",
#     models_dir="lstm_saved_models"
# )
# LSTMTrainer.train_all()

# GRUTrainer = GRUPredictor(
#     data_pkl="data/traffic_model_ready.pkl",
#     models_dir="gru_saved_models"
# )
# GRUTrainer.train_all()

MLPTrainer = MLPPredictor(
    data_pkl="data/traffic_model_ready.pkl",
    models_dir="mlp_saved_models"
)
MLPTrainer.train_all()

# TNCTrainer = TCNPredictor(
#     data_pkl="data/traffic_model_ready.pkl",
#     models_dir="tcn_saved_models"
# )
# TNCTrainer.train_all()