# models/predictor.py

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import TensorDataset, DataLoader
import re

from models.lstm_model import LSTMModel  # relative import
from models.gru_model import GRUModel  # Ensure this is imported
from models.mlp_model import MLPModel
from models.tcn_model import TCNModel


# Configuration
DATA_PKL         = os.path.normpath(os.path.join(os.path.dirname(__file__), '../data/traffic_model_ready.pkl'))
MODELS_DIR_LSTM  = os.path.normpath(os.path.join(os.path.dirname(__file__), 'lstm_saved_models'))
MODELS_DIR_LSTM_2 = os.path.normpath(os.path.join(os.path.dirname(__file__), '../lstm_saved_models'))
MODELS_DIR_GRU   = os.path.normpath(os.path.join(os.path.dirname(__file__), 'gru_saved_models'))
MODELS_DIR_GRU_2 = os.path.normpath(os.path.join(os.path.dirname(__file__), '../gru_saved_models'))
MODELS_DIR_MLP   = os.path.normpath(os.path.join(os.path.dirname(__file__), 'mlp_saved_models'))
MODELS_DIR_MLP_2 = os.path.normpath(os.path.join(os.path.dirname(__file__), '../mlp_saved_models'))
MODELS_DIR_TCN   = os.path.normpath(os.path.join(os.path.dirname(__file__), 'tcn_saved_models'))
MODELS_DIR_TCN_2 = os.path.normpath(os.path.join(os.path.dirname(__file__), '../tcn_saved_models'))
INPUT_DAYS = 1     # history window in days
SEQ_LEN    = 96    # 96 intervals per day (15-min each)



def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)

class LSTMPredictor:
    def __init__(self, data_pkl=DATA_PKL, models_dir=MODELS_DIR_LSTM, models_dir_2=MODELS_DIR_LSTM_2):
        # Load full traffic DataFrame
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.df = pd.read_pickle(data_pkl)
        self.df['Timestamp'] = pd.to_datetime(self.df['Timestamp'])
        # Ensure Site_ID is str for consistency
        self.df['Site_ID'] = self.df['Site_ID'].astype(str)
        # Prepare output directory
        _ensure_dir(models_dir)
        self.models_dir = models_dir
        _ensure_dir(models_dir_2)
        self.models_dir_2 = models_dir_2

    def train_all(self, epochs=5, batch_size=32, lr=1e-3):
        """
        Train & save one LSTM model per (Site_ID, Location) arm.
        """
        grouped = self.df.groupby(['Site_ID','Location'])
        for (site, loc), sub in grouped:
            # Filename: e.g. "0970__WARRIGAL_RD_N_of_HIGH_STREET_RD.pth"
            fname = f"{site}__{loc.replace(' ','_')}.pth"
            out_path = os.path.join(self.models_dir, fname)
            if os.path.exists(out_path):
                continue  # skip already-trained

            # Extract the sorted volume series
            if site == '3001' and loc.upper() == 'CHURCH_ST SW OF BARKERS_RD':
                window = 2 * SEQ_LEN - 2  # 2 days of 15-min intervals
            else:
                window = INPUT_DAYS * SEQ_LEN  # default 7 days


            ts = sub.sort_values('Timestamp')['Volume'].values
            
            if len(ts) < window + 1:
                print(f"⚠ Skipping {site}|{loc}: only {len(ts)} points")
                continue

            # Build sliding windows
            X_list, y_list = [], []
            for i in range(window, len(ts)):
                X_list.append(ts[i-window:i])
                y_list.append(ts[i])
            X_arr = np.stack(X_list, axis=0).astype(np.float32)    # (N, window)
            y_arr = np.array(y_list, dtype=np.float32).reshape(-1,1)  # (N,1)

            # Scale features and targets
            scaler = MinMaxScaler()
            X_flat = X_arr.reshape(-1,1)                   # (N*window,1)
            X_scaled = scaler.fit_transform(X_flat).reshape(-1, window)  # (N,window)
            y_scaled = scaler.transform(y_arr)             # (N,1)

            # Create tensors: (batch, seq_len, 1)
            X_tensor = torch.from_numpy(X_scaled).unsqueeze(-1)  # (N, window, 1)
            y_tensor = torch.from_numpy(y_scaled)                # (N, 1)

            ds = TensorDataset(X_tensor, y_tensor)
            loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

            # Initialize model
            model = LSTMModel(input_size=1, hidden_size=64, num_layers=2).to(self.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            loss_fn = torch.nn.MSELoss()

            # Training loop
            model.train()
            for epoch in range(epochs):
                total_loss = 0.0
                for xb, yb in loader:
                    xb = xb.to(self.device)
                    yb = yb.to(self.device)
                    pred = model(xb)
                    loss = loss_fn(pred, yb)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss = loss.item()
                avg = total_loss / len(loader)
                print(f"[{site}|{loc}] Epoch {epoch+1}/{epochs} ‒ loss: {avg:.4f}")

            # Save model state and scaler
            torch.save({
                'state_dict': model.state_dict(),
                'scaler': scaler
            }, out_path)
            print(f"✔ Saved model: {fname}")

    def predict(self, site, loc, timestamp):
        """
        Predicts traffic volume (vehicles/hour) for a given site & arm at a specific timestamp.
        """
        loc = loc.upper()
        key = f"{site}__{loc.replace(' ','_')}.pth"
        path = os.path.join(self.models_dir_2, key)
        if not os.path.exists(path):
            raise FileNotFoundError(f"No saved model for {site}|{loc}, {key}")

        ckpt = torch.load(path, weights_only = False, map_location=self.device)
        model = LSTMModel(input_size=1, hidden_size=64, num_layers=2).to(self.device)
        model.load_state_dict(ckpt['state_dict'])
        model.eval()
        scaler = ckpt['scaler']

        # Build history window ending just before `timestamp`
        if site == '3001' and loc == 'CHURCH_ST SW OF BARKERS_RD':
            window = 2 * SEQ_LEN - 2
        else:
            window = INPUT_DAYS * SEQ_LEN
        
        ts = pd.to_datetime(timestamp)
        self.df['Location'] = self.df['Location'].str.upper()
        sub = self.df[
            (self.df['Site_ID']==site) &
            (self.df['Location']==loc) &
            (self.df['Timestamp'] < ts)
        ].sort_values('Timestamp').tail(window)
        
        
        

        if len(sub) < window:
            raise ValueError(f"Not enough history for {site}|{loc} at {timestamp} with only {len(sub)} points")

        seq = sub['Volume'].values.astype(np.float32).reshape(-1,1)
        seq_scaled = scaler.transform(seq)                     # (window,1)
        x = torch.from_numpy(seq_scaled).unsqueeze(0)          # (1,window,1)
        x = x.to(self.device)
        with torch.no_grad():
            pred_scaled = model(x).item()
        pred = scaler.inverse_transform([[pred_scaled]])[0][0]
        return float(pred)


class GRUPredictor:
    def __init__(self, data_pkl, models_dir, models_dir_2=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[GRU] Using device: {self.device}")
        
        self.df = pd.read_pickle(data_pkl)
        self.df['Timestamp'] = pd.to_datetime(self.df['Timestamp'])
        self.df['Site_ID'] = self.df['Site_ID'].astype(str)
        self.df['Location'] = self.df['Location'].str.upper()

        self.models_dir = models_dir
        self.models_dir_2 = models_dir_2 or models_dir

        _ensure_dir(self.models_dir)
        _ensure_dir(self.models_dir_2)
    
    def train_all(self, epochs=5, batch_size=32, lr=1e-3):
        """
        Train & save one GRU model per (Site_ID, Location) arm.
        """
        grouped = self.df.groupby(['Site_ID', 'Location'])
        for (site, loc), sub in grouped:
            # Filename: e.g. "0970__WARRIGAL_RD_N_of_HIGH_STREET_RD_GRU.pth"
            fname = f"{site}__{loc.replace(' ', '_')}_GRU.pth"
            out_path = os.path.join(self.models_dir, fname)
            if os.path.exists(out_path):
                continue  # Skip already-trained

            # Extract the sorted volume series
            if site == '3001' and loc.upper() == 'CHURCH_ST SW OF BARKERS_RD':
                window = 2 * SEQ_LEN - 2  # Special case
            else:
                window = INPUT_DAYS * SEQ_LEN  # Default 7 days

            ts = sub.sort_values('Timestamp')['Volume'].values

            if len(ts) < window + 1:
                print(f"⚠ Skipping {site}|{loc}: only {len(ts)} points")
                continue

            # Create sliding windows
            X_list, y_list = [], []
            for i in range(window, len(ts)):
                X_list.append(ts[i - window:i])
                y_list.append(ts[i])
            X_arr = np.stack(X_list, axis=0).astype(np.float32)       # (N, window)
            y_arr = np.array(y_list, dtype=np.float32).reshape(-1, 1) # (N, 1)

            # Normalize using MinMaxScaler
            scaler = MinMaxScaler()
            X_flat = X_arr.reshape(-1, 1)
            X_scaled = scaler.fit_transform(X_flat).reshape(-1, window)
            y_scaled = scaler.transform(y_arr)

            # Convert to tensors: (batch, seq_len, input_size)
            X_tensor = torch.from_numpy(X_scaled).unsqueeze(-1)  # (N, window, 1)
            y_tensor = torch.from_numpy(y_scaled)                # (N, 1)

            dataset = TensorDataset(X_tensor, y_tensor)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            # Build and train GRU model
            model = GRUModel(input_size=1, hidden_size=64, num_layers=2).to(self.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            loss_fn = torch.nn.MSELoss()

            model.train()
            for epoch in range(epochs):
                total_loss = 0.0
                for xb, yb in loader:
                    xb = xb.to(self.device)
                    yb = yb.to(self.device)
                    pred = model(xb)
                    loss = loss_fn(pred, yb)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss = loss.item()
                avg_loss = total_loss / len(loader)
                print(f"[GRU {site}|{loc}] Epoch {epoch+1}/{epochs} ‒ Loss: {avg_loss:.4f}")

            # Save model and scaler
            torch.save({
                'state_dict': model.state_dict(),
                'scaler': scaler
            }, out_path)
            print(f"✔ GRU model saved: {fname}")

    def predict(self, site, loc, timestamp):
        loc = loc.upper()
        fname = f"{site}__{loc.replace(' ','_')}_GRU.pth"
        path = os.path.join(self.models_dir_2, fname)

        if not os.path.exists(path):
            raise FileNotFoundError(f"No saved GRU model for {site}|{loc}, {fname}")

        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        model = GRUModel(input_size=1, hidden_size=64, num_layers=2).to(self.device)
        model.load_state_dict(ckpt['state_dict'])
        model.eval()
        scaler = ckpt['scaler']

        if site == '3001' and loc == 'CHURCH_ST SW OF BARKERS_RD':
            window = 2 * SEQ_LEN - 2
        else:
            window = INPUT_DAYS * SEQ_LEN

        ts = pd.to_datetime(timestamp)
        sub = self.df[
            (self.df['Site_ID'] == site) &
            (self.df['Location'] == loc) &
            (self.df['Timestamp'] < ts)
        ].sort_values('Timestamp').tail(window)

        if len(sub) < window:
            raise ValueError(f"Not enough history for {site}|{loc} at {timestamp}")

        seq = sub['Volume'].values.astype(np.float32).reshape(-1,1)
        seq_scaled = scaler.transform(seq)
        x = torch.from_numpy(seq_scaled).unsqueeze(0).to(self.device)

        with torch.no_grad():
            pred_scaled = model(x).item()
        pred = scaler.inverse_transform([[pred_scaled]])[0][0]
        return float(pred)
    
class MLPPredictor:
    def __init__(self, data_pkl=DATA_PKL,
                       models_dir=MODELS_DIR_MLP,
                       models_dir_2=MODELS_DIR_MLP_2):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[MLP] Using device: {self.device}")
        self.df = pd.read_pickle(data_pkl)
        self.df['Timestamp'] = pd.to_datetime(self.df['Timestamp'])
        self.df['Site_ID'] = self.df['Site_ID'].astype(str)
        self.df['Location'] = self.df['Location'].str.upper()
        _ensure_dir(models_dir);   self.models_dir   = models_dir
        _ensure_dir(models_dir_2); self.models_dir_2 = models_dir_2

    def train_all(self, epochs=5, batch_size=32, lr=1e-3):
        grouped = self.df.groupby(['Site_ID','Location'])
        for (site,loc), sub in grouped:
            fname = f"{site}__{loc.replace(' ','_')}_MLP.pth"
            out_path = os.path.join(self.models_dir, fname)
            if os.path.exists(out_path): continue

            # window same as RNNs
            if site=='3001' and loc=='CHURCH_ST SW OF BARKERS_RD':
                window = 2*SEQ_LEN-2
            else:
                window = INPUT_DAYS*SEQ_LEN

            ts = sub.sort_values('Timestamp')['Volume'].values
            if len(ts) < window + 1:
                print(f"⚠ Skipping {site}|{loc}: only {len(ts)} points"); continue

            # sliding windows
            X_list, y_list = [], []
            for i in range(window, len(ts)):
                X_list.append(ts[i-window:i])
                y_list.append(ts[i])
            X_arr = np.stack(X_list).astype(np.float32)
            y_arr = np.array(y_list, dtype=np.float32).reshape(-1,1)

            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X_arr.reshape(-1,1)).reshape(-1,window)
            y_scaled = scaler.transform(y_arr)

            X_tensor = torch.from_numpy(X_scaled)      # (N,window)
            y_tensor = torch.from_numpy(y_scaled)      # (N,1)
            loader   = DataLoader(TensorDataset(X_tensor,y_tensor),
                                  batch_size=batch_size, shuffle=True)

            model = MLPModel(input_size=window).to(self.device)
            opt   = torch.optim.Adam(model.parameters(), lr=lr)
            loss_fn = torch.nn.MSELoss()
            model.train()
            for ep in range(epochs):
                total=0
                for xb,yb in loader:
                    xb,yb = xb.to(self.device), yb.to(self.device)
                    pred = model(xb)
                    l = loss_fn(pred,yb)
                    opt.zero_grad(); l.backward(); opt.step()
                    total = l.item()
                print(f"[MLP {site}|{loc}] Ep {ep+1}/{epochs} – loss: {total/len(loader):.4f}")

            torch.save({'state_dict':model.state_dict(),'scaler':scaler}, out_path)
            print(f"✔ Saved MLP: {fname}")

    def predict(self, site, loc, timestamp):
        loc = loc.upper()
        key = f"{site}__{loc.replace(' ','_')}_MLP.pth"
        path = os.path.join(self.models_dir_2, key)
        if not os.path.exists(path):
            raise FileNotFoundError(f"No MLP model for {site}|{loc}")

        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        # determine window
        if site=='3001' and loc=='CHURCH_ST SW OF BARKERS_RD':
            window = 2*SEQ_LEN-2
        else:
            window = INPUT_DAYS*SEQ_LEN

        model = MLPModel(input_size=window).to(self.device)
        model.load_state_dict(ckpt['state_dict']); model.eval()
        scaler = ckpt['scaler']

        ts = pd.to_datetime(timestamp)
        hist = (self.df[
                   (self.df['Site_ID']==site)&
                   (self.df['Location']==loc)&
                   (self.df['Timestamp']<ts)
               ]
               .sort_values('Timestamp')['Volume']
               .tail(window)
               .values.astype(np.float32))
        if len(hist)<window:
            raise ValueError(f"Not enough history for {site}|{loc}")

        seq_scaled = scaler.transform(hist.reshape(-1,1)).reshape(1,window)
        x = torch.from_numpy(seq_scaled).to(self.device)
        with torch.no_grad():
            pred_s = model(x).item()
        return float(scaler.inverse_transform([[pred_s]])[0][0])


class TCNPredictor:
    def __init__(self, data_pkl=DATA_PKL,
                       models_dir=MODELS_DIR_TCN,
                       models_dir_2=MODELS_DIR_TCN_2):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[TCN] Using device: {self.device}")
        self.df = pd.read_pickle(data_pkl)
        self.df['Timestamp'] = pd.to_datetime(self.df['Timestamp'])
        self.df['Site_ID'] = self.df['Site_ID'].astype(str)
        self.df['Location'] = self.df['Location'].str.upper()
        _ensure_dir(models_dir);   self.models_dir   = models_dir
        _ensure_dir(models_dir_2); self.models_dir_2 = models_dir_2

    def train_all(self, epochs=5, batch_size=32, lr=1e-3):
        grouped = self.df.groupby(['Site_ID','Location'])
        for (site,loc), sub in grouped:
            fname = f"{site}__{loc.replace(' ','_')}_TCN.pth"
            out_path = os.path.join(self.models_dir, fname)
            if os.path.exists(out_path): continue

            window = 2*SEQ_LEN-2 if (site=='3001' and loc=='CHURCH_ST SW OF BARKERS_RD') else INPUT_DAYS*SEQ_LEN
            ts = sub.sort_values('Timestamp')['Volume'].values
            if len(ts) < window + 1:
                print(f"⚠ Skipping {site}|{loc}: only {len(ts)} points"); continue

            X_list, y_list = [], []
            for i in range(window, len(ts)):
                X_list.append(ts[i-window:i])
                y_list.append(ts[i])
            X_arr = np.stack(X_list).astype(np.float32)
            y_arr = np.array(y_list, dtype=np.float32).reshape(-1,1)

            scaler  = MinMaxScaler()
            Xs      = scaler.fit_transform(X_arr.reshape(-1,1)).reshape(-1,window)
            ys      = scaler.transform(y_arr)
            X_tensor= torch.from_numpy(Xs).unsqueeze(-1)  # (N,window,1)
            y_tensor= torch.from_numpy(ys)
            loader  = DataLoader(TensorDataset(X_tensor,y_tensor),
                                  batch_size=batch_size, shuffle=True)

            model = TCNModel(input_size=1, hidden_size=64, seq_len=window).to(self.device)
            opt   = torch.optim.Adam(model.parameters(), lr=lr)
            loss_fn = torch.nn.MSELoss()
            model.train()
            for ep in range(epochs):
                total=0
                for xb,yb in loader:
                    xb,yb = xb.to(self.device), yb.to(self.device)
                    pred = model(xb)
                    l = loss_fn(pred,yb)
                    opt.zero_grad(); l.backward(); opt.step()
                    total = l.item()
                print(f"[TCN {site}|{loc}] Ep {ep+1}/{epochs} – loss: {total/len(loader):.4f}")

            torch.save({'state_dict':model.state_dict(),'scaler':scaler}, out_path)
            print(f"✔ Saved TCN: {fname}")

    def predict(self, site, loc, timestamp):
        loc = loc.upper()
        key = f"{site}__{loc.replace(' ','_')}_TCN.pth"
        path = os.path.join(self.models_dir_2, key)
        if not os.path.exists(path):
            raise FileNotFoundError(f"No TCN model for {site}|{loc}")

        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        window = 2*SEQ_LEN-2 if (site=='3001' and loc=='CHURCH_ST SW OF BARKERS_RD') else INPUT_DAYS*SEQ_LEN

        model = TCNModel(input_size=1, hidden_size=64, seq_len=window).to(self.device)
        model.load_state_dict(ckpt['state_dict']); model.eval()
        scaler = ckpt['scaler']

        seq = (self.df[
                   (self.df['Site_ID']==site)&
                   (self.df['Location']==loc)&
                   (self.df['Timestamp']<pd.to_datetime(timestamp))
               ]
               .sort_values('Timestamp')['Volume']
               .tail(window)
               .values.astype(np.float32)
           ).reshape(-1,1)
        if len(seq)<window:
            raise ValueError(f"Not enough history for {site}|{loc}")

        seq_scaled = scaler.transform(seq)
        x = torch.from_numpy(seq_scaled).unsqueeze(0).to(self.device)  # (1,window,1)
        with torch.no_grad():
            pred_s = model(x).item()
        return float(scaler.inverse_transform([[pred_s]])[0][0])