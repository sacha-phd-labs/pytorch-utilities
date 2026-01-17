import torch
import abc
import mlflow
import os

class PytorchTrainer:

    def __init__(self, metrics=[]):

        # device
        self.device = self.get_device()

        # create dataset / loader / model / optimizer / objective / metrics
        self.loader_train, self.loader_val = self.create_data_loader()
        self.model = self.create_model()
        self.model = self.model.to(self.device)
        self.signature = self.get_signature()
        self.optimizer = self.get_optimizer(learning_rate=self.learning_rate)
        self.objective = self.get_objective()
        self.metrics = self.get_metrics(metrics)

        #
        self.initial_epoch = 0

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
        return out

    def reset_metrics(self):
        for metric in self.metrics:
            metric.reset_states()

    def on_epoch_end(self, epoch):
        for metric in self.metrics:
            metric.reset_states()

    def update_metrics(self, loss, y_true, y_pred):
        for metric in self.metrics:
            if 'loss' in metric.name:
                metric.update_state(None, loss)
            else:
                metric.update_state(y_true, y_pred)

    def load_model_and_optimizer(self, artifact_path):
        """Properly load model and optimizer state from mlflow artifacts."""
        #
        try:
            mlflow.artifacts.download_artifacts(artifact_path=artifact_path, dst_path="/tmp", run_id=mlflow.active_run().info.run_id)
        except mlflow.exceptions.MlflowException:
            print("No reboot model found in mlflow artifacts.")
            return
        #
        local_model_path = f"/tmp/{artifact_path}/model.pth"
        local_epoch_path = f"/tmp/{artifact_path}/epoch.txt"
        local_optimizer_path = f"/tmp/{artifact_path}/optimizer.pth"
        if os.path.exists(local_model_path) and os.path.exists(local_epoch_path) and os.path.exists(local_optimizer_path):
            self.model.load_state_dict(torch.load(local_model_path))
            with open(local_epoch_path, "r") as f:
                self.initial_epoch = int(f.read()) + 1
            self.optimizer.load_state_dict(torch.load(local_optimizer_path))
            print(f"Rebooted model from epoch {self.initial_epoch}")
        else:
            print("No reboot model found in mlflow artifacts.")

    def mlflow_log_model_as_artifact(self, epoch, artifact_path):

        # log reboot model as artifact
        os.makedirs(f"/tmp/{artifact_path}", exist_ok=True)
        torch.save(self.model.state_dict(), f"/tmp/{artifact_path}/model.pth")
        with open(f"/tmp/{artifact_path}/epoch.txt", "w") as f:
            f.write(str(epoch))
        mlflow.log_artifact(f"/tmp/{artifact_path}/model.pth", artifact_path=artifact_path)
        mlflow.log_artifact(f"/tmp/{artifact_path}/epoch.txt", artifact_path=artifact_path)
        # save optimizer state
        torch.save(self.optimizer.state_dict(), f"/tmp/{artifact_path}/optimizer.pth")
        mlflow.log_artifact(f"/tmp/{artifact_path}/optimizer.pth", artifact_path=artifact_path)

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
            self.mlflow_log_model_as_artifact(epoch, artifact_path=f"best_model_{metric_name}")
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