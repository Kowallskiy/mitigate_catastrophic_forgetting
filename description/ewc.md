# Elastic Weight Consolidation (EWC)

## Introduction

Elastic Weight Consolidation (EWC) is a technique developed to combat the problem of catastrophic forgetting in neural networks. When a neural network is trained sequentially on multiple tasks, it tends to forget the knowledge acquired from previous tasks. EWC addresses this issue by slowing down learning on certain weights based on their importance to previous tasks, allowing the network to retain previously learned knowledge while acquiring new information.

This project implements the EWC technique, demonstrating its efficancy in enabling continual learning in neural networks.

## Implementation

### Neural Network Architecture

After preprocessing the data, we define a straightforward neural network architecture. The network includes two hidden layers and a final classification layer, with ReLU activations following each hidden layer.

```Python
# Define a simple feedforward neural network
class NeuralNetwork(nn.Module):
    def __init__(self, num_classes=10, hidden_size=512):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()

        # Input layer
        self.fc1 = nn.Linear(28*28, hidden_size)
        self.relu1 = nn.ReLU()
        # Hidden layer
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        # Output layer
        self.classifier = nn.Linear(hidden_size, num_classes)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # He initialization for weights in linear layers
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        logits = self.classifier(x)

        return logits

```

### Training on Task 1

The model was initially trained on Task 1, which involves the first six labels of the FashionMNIST dataset, achieving 90% accuracy. Next, we compute the Fisher information matrix to prepare for the EWC implementation.

### Calculating the Fisher Matrix

Catastrophic forgetting occurs because the neural network's weights are updated to minimize the error on the new task, often at the expense of performance on previously learned tasks. EWC mitigates this by adding a regularization term to the loss function, which penalizes changes to important weights.

The regularization term is derived from the Fisher information matrix, which measures the sensitivity of the output distribution to changes in the parameters (weights). The EWC loss function is given by:

```math
L(\theta) = L_T(\theta) + \sum_i \frac{\lambda}{2} F_i (\theta_i - \theta_{i,old})^2
```
where:
* $L(\theta)$ is the total loss function with EWC regularization,
* $L_T(\theta)$ is the loss on the current task
* $\lambda$ is a hyperparameter that controls the strength of the regularization
* $F_i$ represents the Fisher information of parameter $\theta_{i}$
* $\theta_{i}$ is the current value of parameter $i$
* and $\theta_{i,old}$ is the value of parameter $i$ after the previous task was learned.

EWC adds a regularization term to the loss function to penalize significant changes to weights that are important for previously learned tasks. The Fisher information matrix quantifies the importance of each weight.

```Python
# Compute the diagonal of the Fisher information matrix for all model parameters
def get_fisher_diag(model, dataloader, params, empirical=False):
    fisher = {}
    params_dict = dict(params)

    for n, p in deepcopy(params_dict).items():
        p.data.zero_()
        fisher[n] = p.data.clone().detach().requires_grad_()

    model.eval()

    for input, gt_label in dataloader:
        input, gt_label = input.to(device), gt_label.to(device)
        model.zero_grad()
        output = model(input)

        # Use empirical Fisher information or true labels based on the flag
        label = output.max(1)[1] if empirical else gt_label

        negloglikelihood = torch.nn.functional.nll_loss(torch.nn.functional.log_softmax(output, dim=1), label)
        negloglikelihood.backward()

        for n, p in model.named_parameters():
            # Accumulate squared gradients
            fisher[n].data += p.grad.data ** 2 / len(dataloader.dataset)

    return {n: p for n, p in fisher.items()}

# Compute the EWC loss, adding it to the traditional loss function
def get_ewc_loss(model, fisher, p_old):
    loss = 0
    for n, p in model.named_parameters():
        # Calculate EWC loss component for each parameter
        _loss = fisher[n] * (p - p_old[n]) ** 2
        loss += _loss.sum()
    return loss

```

### Training on Task 2

We then apply the EWC mechanism during the training on Task 2.

```Python
ewc_lambda = 1_000_000  # EWC regularization strength

# Compute Fisher matrix after training on Task 1
fisher_matrix = get_fisher_diag(model_task1, train_dataloader_task1, model_task1.named_parameters())
# Save parameters after Task 1 for EWC
prev_params = {n: p.data.clone() for n, p in model_task2.named_parameters()}
```
### Incorporating EWC into the Training Loop

The training loop for Task 2 integrates the EWC loss with the standard cross-entropy loss.

```Python
# Train model considering both the original loss and EWC loss
def train(dataloader, model, loss_fn, optimizer, fisher_matrix, prev_params):
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        ce_loss = loss_fn(pred, y)  # Cross-entropy loss

        ewc_loss = get_ewc_loss(model, fisher_matrix, prev_params)  # EWC loss
        loss = ce_loss + ewc_lambda * ewc_loss  # Total loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            print(f"Loss: {loss.item():>7f} [{batch * len(X):>5d}/{len(dataloader.dataset):>5d}]")
```

### Evaluation and Adjustments

Despite the initial success on Task 1, the model's accuracy on the task dropped to 21% after training on Task 2, suggesting inadequate regularization strength. Adjusting the __ewc_lambda__ showed some improvement, yet highlighted the delicate balance required in tuning EWC's regularization to prevent overfitting or underfitting.

### Conclusion

This revised implementation guide introduces a structured approach to incorporating EWC into a neural network model, detailing the network architecture, the process for calculating the Fisher information matrix, and the integration of EWC regularization into the training process. Fine-tuning the EWC hyperparameters, particularly __ewc_lambda__, is crucial for balancing knowledge retention and acquisition in continual learning scenarios.
