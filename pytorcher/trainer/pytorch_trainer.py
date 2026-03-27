import torch
import abc
import mlflow
import os
import numpy as np
import random

class PytorchTrainer:

    def __init__(self, metrics=[], seed=42, **kwargs):

        # device
        self.device = self.get_device()

        # create dataset / loader / model / optimizer / objective / metrics
        self.loader_train, self.loader_val = self.create_data_loader()
        self.model = self.create_model()
        self.model = self.model.to(self.device)
        self.signature = self.get_signature()
        self.optimizer = self.get_optimizer()
        self.objective = self.get_objective()
        self.metrics = self.get_metrics(metrics)

        #
        self.initial_epoch = 0
        #
        self.set_seed(seed)

    def set_seed(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    def get_device(self):
        # Create device for training (GPU if available)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {self.device}')
        return self.device

    @abc.abstractmethod
    def create_data_loader(self):
        """
        self.dataset = SomeDataset(params...)
        loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=self.shuffle)

        return loader
        """
        pass

    def get_signature(self):
        sample = self.dataset.__getitem__(0)[0]
        signature = (1, ) + tuple(sample.shape)
        return signature

    @abc.abstractmethod
    def create_model(self):
        """
        model = someModel(params...)
        model = model.to(self.device)
        return model
        """
        pass

    def get_optimizer(self, learning_rate=1e-3):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        return optimizer

    @abc.abstractmethod
    def get_objective(self):
        """
        objective = someLossFunction()
        return objective
        """
        pass

    def get_metrics(self, metrics):
        out = []
        for metric in metrics:
            metric_name, metric_config = metric
            metric_class = getattr(__import__('pytorcher.metrics', fromlist=[metric_name]), metric_name)
            out.append(metric_class(**metric_config))
        if not any('loss' in m.name for m in out):
            out.append(getattr(__import__('pytorcher.metrics', fromlist=[metric_name]), 'Mean')(**{'name': 'loss'}))
        self.metrics = out
        return self.metrics

    def reset_metrics(self):
        for metric in self.metrics:
            metric.reset_states()

    def on_epoch_end(self, epoch):
        for metric in self.metrics:
            metric.reset_states()

    def update_metrics(self, y_true, y_pred):
        for metric in self.metrics:
            if 'loss' not in metric.name:
                metric.update_state(y_true, y_pred)

    def update_loss(self, loss):
        for metric in self.metrics:
            if 'loss' in metric.name:
                metric.update_state(None, loss)

    def load_checkpoint(self, artifact_path):
        """Properly load model, optimizer, and RNG state from mlflow artifacts."""

        try:
            mlflow.artifacts.download_artifacts(
                artifact_path=artifact_path,
                dst_path="/tmp",
                run_id=mlflow.active_run().info.run_id,
            )
        except mlflow.exceptions.MlflowException:
            print(f"No model found in mlflow artifacts at {artifact_path}")
            return

        checkpoint_path = f"/tmp/{artifact_path}/checkpoint.pth"
        if not os.path.exists(checkpoint_path):
            print(f"No model found in mlflow artifacts at {artifact_path}")
            return
        else:
            checkpoint = torch.load(checkpoint_path, weights_only=False) # weights_only False allows to load numpy and python RNG states as well, not only torch RNG state

            # Load model
            self.model.load_state_dict(checkpoint["model"])

            # Load optimizer
            self.optimizer.load_state_dict(checkpoint["optimizer"])

            # Load epoch
            self.initial_epoch = checkpoint["epoch"] + 1

            # Restore RNG states
            torch.set_rng_state(checkpoint["torch_rng_state"])

            if torch.cuda.is_available() and checkpoint["cuda_rng_state"] is not None:
                torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])

            np.random.set_state(checkpoint["numpy_rng_state"])
            random.setstate(checkpoint["python_rng_state"])

            if 'dataloader_rng_state' in checkpoint and self.loader_train is not None and hasattr(self.loader_train, 'generator') and self.loader_train.generator is not None:
                self.loader_train.generator.set_state(checkpoint["dataloader_rng_state"])

    def mlflow_log_checkpoint_as_artifact(self, epoch, artifact_path):

        os.makedirs(f"/tmp/{artifact_path}", exist_ok=True)
        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": epoch,

        }

        # Save RNG states
        rng_state = {
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy_rng_state": np.random.get_state(),
            "python_rng_state": random.getstate(),
        }
        if self.loader_train is not None and hasattr(self.loader_train, 'generator') and self.loader_train.generator is not None:
            rng_state["dataloader_rng_state"] = self.loader_train.generator.get_state()
            #
        checkpoint.update(rng_state)

        #
        torch.save(checkpoint, f"/tmp/{artifact_path}/checkpoint.pth")

        #
        mlflow.log_artifact(f"/tmp/{artifact_path}/checkpoint.pth", artifact_path=artifact_path)

    def mlflow_metric_monitoring(self, epoch, metric_name, current_metric_value, mode='max'):
        """
        Log model if target metric improved.
        """

        client = mlflow.client.MlflowClient()
        best_value = client.get_metric_history(mlflow.active_run().info.run_id, metric_name)
        if mode == 'max':
            condition = len(best_value) == 0 or current_metric_value > max([m.value for m in best_value])
        elif mode == 'min':
            condition = len(best_value) == 0 or current_metric_value < min([m.value for m in best_value])
        else:
            raise ValueError(f"Unknown mode {mode} for metric monitoring.")
        if condition:
            # log model as artifact
            self.mlflow_log_checkpoint_as_artifact(epoch, artifact_path=f"best_model_{metric_name}")
            print(f'New best {metric_name}={current_metric_value:.4f} at epoch {epoch+1}, model logged.')


    def fit(self):

        for epoch in range(self.initial_epoch, self.n_epochs):
            self.model.train()
            epoch_loss = 0.0
            for batch_idx, (inputs, targets) in enumerate(self.loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.objective(outputs, targets)
                loss.backward()
                self.optimizer.step()

                self.update_metrics(loss.item(), targets, outputs)

            print(f'Epoch {epoch+1}, metrics: ')
            for metric in self.metrics:
                print(f'{metric.name}: {metric.result():.4f}')
                metric.reset_states()

            #
            self.on_epoch_end(epoch)