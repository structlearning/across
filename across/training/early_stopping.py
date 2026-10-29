import os

import torch


class EarlyStoppingModule:
    """
    Module to keep track of validation score across epochs.
    Stop training if score not improving exceeds patience.

    Directory layout:
        <save_dir>/initialModels/<run_id>/initial.pt   — untrained starting state
        <save_dir>/latestModels/<run_id>/<idx>.pt      — indexed checkpoint per validation call
        <save_dir>/<best_val_name>/<run_id>/best.pt    — best checkpoint so far

    To re-run a config from scratch, move or delete the three <run_id> subdirectories.
    """

    def __init__(self, save_dir, run_id, patience, delta, logger, best_val_name, full_run_id=None):
        self.save_dir = save_dir
        self.run_id = run_id
        self.full_run_id = full_run_id if full_run_id is not None else run_id
        self.patience = patience
        self.delta = delta
        self.logger = logger
        self.best_val_name = best_val_name
        self.create_dirs()
        self.best_scores = None
        self.num_bad_epochs = 0
        self.should_stop_now = False
        self._save_count = 0

    def create_dirs(self):
        initial_dir = os.path.join(self.save_dir, "initialModels", self.run_id)
        os.makedirs(initial_dir, exist_ok=True)
        self.initial_model_path = os.path.join(initial_dir, "initial.pt")

        self.latest_model_dir = os.path.join(self.save_dir, "latestModels", self.run_id)
        os.makedirs(self.latest_model_dir, exist_ok=True)

        best_dir = os.path.join(self.save_dir, self.best_val_name, self.run_id)
        os.makedirs(best_dir, exist_ok=True)
        self.best_model_path = os.path.join(best_dir, "best.pt")

        for d in [initial_dir, self.latest_model_dir, best_dir]:
            with open(os.path.join(d, "task_name.txt"), "w") as f:
                f.write(self.full_run_id)

    def save_initial_state(self, state_dict):
        #  assert not os.path.exists(self.initial_model_path), \
        #      f"{self.initial_model_path} already exists. Do you mean to resume?"
        self.logger.info(f"saving initial model to {self.initial_model_path}")
        torch.save(state_dict, self.initial_model_path)

    def save_latest_model(self, state_dict):
        path = os.path.join(self.latest_model_dir, f"{self._save_count}.pt")
        torch.save(state_dict, path)
        self._save_count += 1

    def load_latest_model(self):
        checkpoints = [f for f in os.listdir(self.latest_model_dir) if f.endswith(".pt")]
        assert checkpoints, \
            f"No checkpoints found in {self.latest_model_dir}. Do you mean to start fresh?"
        latest_file = max(checkpoints, key=lambda f: int(f[:-3]))
        path = os.path.join(self.latest_model_dir, latest_file)
        self.logger.info(f"loading latest trained model from {path}")
        checkpoint = torch.load(path)
        self.patience = checkpoint["patience"]
        self.best_scores = checkpoint["best_scores"]
        self.num_bad_epochs = checkpoint["num_bad_epochs"]
        self.should_stop_now = checkpoint["should_stop_now"]
        self._save_count = int(latest_file[:-3]) + 1
        return checkpoint

    def save_best_model(self, state_dict):
        self.logger.info(f"saving best validated model to {self.best_model_path}")
        torch.save(state_dict, self.best_model_path)

    def load_best_model(self, device="cuda"):
        self.logger.info(f"loading best validated model from {self.best_model_path}")
        checkpoint = torch.load(self.best_model_path, map_location=device)
        return checkpoint

    def diff(self, curr_scores):
        return sum([cs - bs for cs, bs in zip(curr_scores, self.best_scores)])

    def check(self, curr_scores, state_dict):
        if self.best_scores is None:
            self.best_scores = curr_scores
            if "best_val_ap" in state_dict:
                state_dict["best_val_ap"] = state_dict["val_ap_score"]
            if "best_val_map" in state_dict:
                state_dict["best_val_map"] = state_dict["val_map_score"]
            if "best_neg_val_loss" in state_dict:
                state_dict["best_neg_val_loss"] = state_dict["neg_val_loss"]
            state_dict["best_scores"] = self.best_scores
            state_dict["num_bad_epochs"] = self.num_bad_epochs
            self.save_best_model(state_dict)
        elif self.diff(curr_scores) >= self.delta:
            self.num_bad_epochs = 0
            self.best_scores = curr_scores
            if "best_val_ap" in state_dict:
                state_dict["best_val_ap"] = state_dict["val_ap_score"]
            if "best_val_map" in state_dict:
                state_dict["best_val_map"] = state_dict["val_map_score"]
            if "best_neg_val_loss" in state_dict:
                state_dict["best_neg_val_loss"] = state_dict["neg_val_loss"]
            state_dict["best_scores"] = self.best_scores
            state_dict["num_bad_epochs"] = self.num_bad_epochs
            self.save_best_model(state_dict)
        else:
            self.num_bad_epochs += 1
            if self.num_bad_epochs > self.patience:
                self.should_stop_now = True
        self.save_latest_model(state_dict)
        return state_dict
