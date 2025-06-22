import torch
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

torch.manual_seed(42)

class nn_wrapper:
    class neural_network(torch.nn.Module):
        def __init__(self, input_size: int, output_size: int):
            """
            Feed-forward neural network for classification.
            """
            super().__init__()
            self.fc1 = torch.nn.Linear(input_size, 512)
            self.bn1 = torch.nn.BatchNorm1d(512)
            self.dropout1 = torch.nn.Dropout(0.3)

            self.fc2 = torch.nn.Linear(512, 1024)
            self.bn2 = torch.nn.BatchNorm1d(1024)
            self.dropout2 = torch.nn.Dropout(0.4)

            self.fc3 = torch.nn.Linear(1024, 512)
            self.bn3 = torch.nn.BatchNorm1d(512)
            self.dropout3 = torch.nn.Dropout(0.3)

            self.fc4 = torch.nn.Linear(512, 128)
            self.bn4 = torch.nn.BatchNorm1d(128)
            self.dropout4 = torch.nn.Dropout(0.2)

            self.fc5 = torch.nn.Linear(128, output_size)
            self.relu = torch.nn.ReLU()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Forward pass of the neural network.
            """
            x = self.fc1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.dropout1(x)

            x = self.fc2(x)
            x = self.bn2(x)
            x = self.relu(x)
            x = self.dropout2(x)

            x = self.fc3(x)
            x = self.bn3(x)
            x = self.relu(x)
            x = self.dropout3(x)

            x = self.fc4(x)
            x = self.bn4(x)
            x = self.relu(x)
            x = self.dropout4(x)

            x = self.fc5(x)
            return x

    def __init__(self, train_data: pd.DataFrame, test_data: pd.DataFrame):
        """
        Initializes the wrapper, splits data, and sets up the model, loss, and optimizer.
        """
        self.device = self.check_device()

        self.train_data = train_data
        self.columns = train_data.columns
        self.test_data = test_data
        self.test_data_index = test_data.index
        self.test_data = self.test_data.apply(pd.to_numeric, errors='raise')
        self.test_data = torch.tensor(test_data.values, dtype=torch.float32)

        if self.device != "cpu":
            self.test_data = self.test_data.to(self.device)
        
        self.percentage = 0.8  # Percentage of data to use for training
        self.X_train, self.y_train, self.X_validation, self.y_validation = self.train_val_split()

        self.train_dataset = TensorDataset(self.X_train, self.y_train)
        self.validation_dataset = TensorDataset(self.X_validation, self.y_validation)

        self.batch_size = 64
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False)
        self.validation_loader = DataLoader(self.validation_dataset, batch_size=self.batch_size)

        input_size = self.X_train.shape[1]
        output_size = torch.unique(self.y_train).numel()

        self.model = self.neural_network(input_size, output_size).to(self.device)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=0.015)

        self.train_history = []
        self.validation_history = []

    def check_device(self) -> str:
        """
        Checks and returns the available device: 'cuda', 'mps', or 'cpu'.
        """
        if torch.cuda.is_available():
            device = "cuda" # NVIDIA GPU
        elif torch.backends.mps.is_available():
            device = "mps" # Apple GPU
        else:
            device = "cpu" # Defaults to CPU if NVIDIA GPU/Apple GPU aren't available
        print(f"Using device: {device}")
        return device

    def train_val_split(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Shuffles and splits the training data into training and validation sets.
        Returns:
            X_train, y_train, X_validation, y_validation (all as torch.Tensors)
        """

        length = self.train_data.shape[0]
        shuffled_data = self.train_data.sample(frac=1).reset_index(drop=True)

        train_samples = int(length * self.percentage)

        train_data = shuffled_data[:train_samples]
        validation_data = shuffled_data[train_samples:]

        # Assume last column is the label
        X_train = train_data.iloc[:, :-1]
        y_train = train_data.iloc[:, -1]
        X_validation = validation_data.iloc[:, :-1]
        y_validation = validation_data.iloc[:, -1]

        X_train = torch.tensor(X_train.values, dtype=torch.float32)
        y_train = torch.tensor(y_train.values, dtype=torch.long)
        X_validation = torch.tensor(X_validation.values, dtype=torch.float32)
        y_validation = torch.tensor(y_validation.values, dtype=torch.long)

        if self.device != "cpu":
            X_train = X_train.to(self.device)
            X_validation = X_validation.to(self.device)
            y_train = y_train.to(self.device)
            y_validation = y_validation.to(self.device)

        return X_train, y_train, X_validation, y_validation 

    def train(self, epochs: int = 15, plot=False) -> None:
        """
        Trains the neural network for a given number of epochs using mini-batches.
        """
        print("Training...")       

        for epoch in range(epochs):
            self.model.train()
            train_loss_epoch = 0.0
            for X_batch, y_batch in self.train_loader:
                y_pred = self.model(X_batch)
                train_loss = self.loss_fn(y_pred, y_batch)
                self.optimizer.zero_grad()
                train_loss.backward()
                self.optimizer.step()
                train_loss_epoch += train_loss.item() * X_batch.size(0)

            train_loss_epoch /= len(self.train_dataset)

            # Validation
            self.model.eval()
            validation_loss_epoch = 0.0
            with torch.inference_mode():
                for X_batch, y_batch in self.validation_loader:
                    validation_pred = self.model(X_batch)
                    validation_loss = self.loss_fn(validation_pred, y_batch)
                    validation_loss_epoch += validation_loss.item() * X_batch.size(0)
            validation_loss_epoch /= len(self.validation_dataset)

            self.train_history.append(train_loss_epoch)
            self.validation_history.append(validation_loss_epoch)

            print(f"Epoch: {epoch + 1} | Train loss: {train_loss_epoch:.4f} | validation loss: {validation_loss_epoch:.4f}")

        if plot:
            plt.figure(figsize=(8, 5))
            plt.plot(self.train_history, label="Train Loss")
            plt.plot(self.validation_history, label="Validation Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Training and Validation Loss")
            plt.legend()
            plt.show()

    def calculate_metrics(self) -> dict[str, float]:
        """
        Calculates and prints accuracy, precision, recall, and F1-score on the validation set.
        Returns:
            Dictionary with metric names and values.
        """
        self.model.eval()
        with torch.inference_mode():
            y_pred = self.model(self.X_validation)
            _, predicted = torch.max(y_pred, 1)
            y_true = self.y_validation.cpu().numpy()
            y_pred_np = predicted.cpu().numpy()

            accuracy = accuracy_score(y_true, y_pred_np)
            precision = precision_score(y_true, y_pred_np, average='weighted', zero_division=0)
            recall = recall_score(y_true, y_pred_np, average='weighted', zero_division=0)
            f1 = f1_score(y_true, y_pred_np, average='weighted', zero_division=0)

            print("| Metric    | Value  |")
            print("|-----------|--------|")
            print(f"| Accuracy  | {accuracy:.4f} |")
            print(f"| Precision | {precision:.4f} |")
            print(f"| Recall    | {recall:.4f} |")
            print(f"| F1-score  | {f1:.4f} |")

            return {
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1)
            }
        
    def predict(self) -> pd.Series:
        """
        Predicts class labels for the test data.
        Returns:
            pd.Series with predictions, indexed by the original test data index.
        """
        self.model.eval()
        with torch.inference_mode():
            y_pred = self.model(self.test_data)
            _, topk_indices = torch.topk(y_pred, k=3, dim=1)  
            topk_str = [' '.join(map(str, row)) for row in topk_indices.cpu().numpy()]
        return pd.Series(topk_str, index=self.test_data_index, name="Prediction")       
    
    def save_model(self, path: str) -> None:
        """
        Saves the trained model to the specified path.
        Args:
            path (str): Path where the model will be saved.
        """
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")