import copy
import json
import os
from datetime import datetime

import torch


class Logger:
    def __init__(self, experiment_name, save_root="./results"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = os.path.join(save_root, experiment_name, timestamp)
        os.makedirs(self.save_dir, exist_ok=True)

        print(f"[OK] Experiment output directory: {self.save_dir}")

        self.start_time = datetime.now()
        self.history = {
            "loss": [],
            "loss_components": {},
            "params": {},
            "epoch": [],
            "time": [],
        }

    def log_config(self, config_dict):
        config_to_save = copy.deepcopy(config_dict)
        env_seed = os.environ.get("PINN_SEED")
        resolved_seed = (
            config_to_save.get("seed")
            or config_to_save.get("training", {}).get("seed")
            or config_to_save.get("runtime", {}).get("seed")
            or env_seed
        )
        if resolved_seed is not None:
            config_to_save["seed"] = resolved_seed
            config_to_save.setdefault("training", {})["seed"] = resolved_seed
            config_to_save.setdefault("runtime", {})["seed"] = resolved_seed

        config_path = os.path.join(self.save_dir, "config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_to_save, f, indent=4, ensure_ascii=False)
        print(f"[INFO] Config saved: {config_path}")

        if resolved_seed is not None:
            seed_file = os.path.join(self.save_dir, f"seed_{resolved_seed}.txt")
            with open(seed_file, "w", encoding="utf-8") as f:
                f.write(f"This run used SEED: {resolved_seed}\n")
            print(f"[INFO] Seed saved: {seed_file}")

    def log_metrics(self, epoch, metrics, time_sec=None):
        self.history["epoch"].append(epoch)
        self.history["loss"].append(metrics["loss"])
        self.history["time"].append(0.0 if time_sec is None else time_sec)

        for key, value in metrics["components"].items():
            self.history["loss_components"].setdefault(key, []).append(value)

        for key, value in metrics["params"].items():
            self.history["params"].setdefault(key, []).append(value)

    def save_history(self):
        duration = datetime.now() - self.start_time
        self.history["duration_seconds"] = duration.total_seconds()
        self.history["duration_str"] = str(duration).split(".")[0]

        save_path = os.path.join(self.save_dir, "history.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=4)
        print(f"[INFO] History saved: {save_path} (duration: {self.history['duration_str']})")

    def save_model(self, model, filename="model_best.pth"):
        save_path = os.path.join(self.save_dir, filename)
        torch.save(model.state_dict(), save_path)

    def get_save_dir(self):
        return self.save_dir
