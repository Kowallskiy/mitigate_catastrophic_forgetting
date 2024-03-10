# Synaptic Intelligence (SI)

## Introduction

Synaptic Intelligence (SI) offers a novel approach to mitigating catastrophic forgetting in neural networks. It achieves this by quantifying the importance of each parameter for past tasks and penalizing changes to these crucial parameters during the learning of new tasks.

## Implementation

### Neural Network Architecture

This segment defines a simple fully connected neural network with two hidden layers. This architecture is used for classification tasks, suitable for datasets like MNIST.

```Python
class NeuralNetwork(nn.Module):
    def __init__(self, num_classes=10, hidden_size=512):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.classifier = nn.Linear(hidden_size, num_classes)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        logits = self.classifier(x)

        return logits
```

### Setting Up Parameters for SI

Initializes the omega (W) parameters that track the importance of each weight and old parameters (p_old) for computing the surrogate loss, key components in the SI method.

```Python
# Setting up Omega parameter
W = {n: torch.zeros_like(p, requires_grad=False) for n, p in model.named_parameters() if p.requires_grad}
# Setting up old parameters for computing surrogate loss in the future
p_old = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}
```

### Surrogate Loss Function

Defines the surrogate loss function for SI, calculating the penalty based on changes to important weights since the last task.

$$
\mathcal{L}_{\text{SI}} = \sum_{i} \Omega_{i} ( \theta_{i} - \theta_{i, \text{old}} )^2
$$

where:
* ${L}_{\text{SI}}$ is the surrogate loss for Synaptic Intelligence
* $\Omega_{i}$ represents the importance of parameter $i$, accumulated over tasks
* $\theta_{i}$ is the current value of parameter $i$
* $\theta_{i, \text{old}}$ is the value of parameter $i$ before learning the current task.

```Python
# Surrogate loss function for Synaptic Intelligence
def surrogate_loss(W, p_old):
    loss = 0
    for n, p in model.named_parameters():
        if p.requires_grad:
            loss += (W[n] * (p - p_old[n]) ** 2).sum()
    return loss
```

### Updating Omega (Importance of Parameters)

Updates the omega parameters after each task, reflecting the importance of each parameter based on how much they contributed to the performance on the current task.

$$
\Omega_{i} \leftarrow \Omega_{i} + \frac{\partial \mathcal{L}_{\text{task}}}{\partial \theta_{i}} \Big|_{\theta_{i} = \theta_{i,\text{old}}} \cdot (\theta_{i} - \theta_{i,\text{old}})
$$

This formula captures the sensitivity of the loss to changes in each parameter, providing a measure of its importance to the learned task.

```Python
# Update omega after completing a task
def update_omega(W, param_importance, epsilon=0.1):
    for n, p in model.named_parameters():
        if p.requires_grad:
            delta = p.detach() - p_old[n]
            W[n] += param_importance[n] / (delta ** 2 + epsilon)
```

### Training with SI

The training loop incorporates SI by dynamically updating parameter importance, applying the SI regularization through the surrogate loss, and integrating it with the task-specific loss.

```Python
# Synaptic Intelligence regularization coefficient
c = 0.6

# Training loop
for i, task_data in enumerate([train_dataloader_task1, train_dataloader_task2]):
    param_importance = {n: torch.zeros_like(p, requires_grad=False) for n, p in model.named_parameters() if p.requires_grad}

    for epoch in range(3):
        for X, y in task_data:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)

            # Compute gradients for the current task
            loss.backward(retain_graph=True)

            # Update parameter importance dynamically during training
            for n, p in model.named_parameters():
                if p.requires_grad:
                    if p.grad is not None:
                        param_importance[n] += p.grad.abs() * (p.detach() - p_old[n]).abs()

            # Apply Synaptic Intelligence regularization
            si_loss = surrogate_loss(W, p_old)
            total_loss = loss + c * si_loss

            # Backward pass on total loss
            total_loss.backward()

            optimizer.step()

        # Evaluation after each epoch
        if i == 0:  # For task 1
            val_accuracy = evaluate_model(model, test_dataloader_task1, device)
        else:  # For task 2
            val_accuracy = evaluate_model(model, test_dataloader_task2, device)
        print(f'Epoch {epoch+1}, Task {i+1}, Validation Accuracy: {val_accuracy:.2f}')

    # Update omega (W) for the next task after training is complete
    update_omega(W, param_importance)

    # Update old parameters (p_old) for the next task
    p_old = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}
```

### Evaluation and Adjustments
After training the neural network with Synaptic Intelligence on sequential tasks, I dedicated time to thoroughly evaluate the model's performance. This evaluation focused on the model's accuracy across different tasks, specifically looking at how well it retained knowledge from the first task after being trained on the second. To achieve this, I used a validation dataset corresponding to each task and measured the model's accuracy.

The accuracy on the first task after training on the second task served as a critical metric for assessing catastrophic forgetting, a common problem where learning new information leads to a significant degradation of performance on previously learned tasks. Despite the model's ability to learn the second task, I observed that the accuracy on the first task dropped to 13%, indicating a substantial level of forgetting.

Recognizing the need for adjustments, my attention turned to the Synaptic Intelligence regularization coefficient, denoted as c in the code. This coefficient plays a pivotal role in balancing between learning new tasks and retaining information from previous tasks. It controls the strength of the regularization term that penalizes changes to weights that are important for the previous tasks.

To address the forgetting issue, I experimented with different values of the c coefficient. The goal was to find an optimal balance where the model could effectively learn new tasks without significantly compromising its performance on the old ones. This process involved trial and error, gradually adjusting the coefficient's value, and observing the impact on the model's accuracy for both the new and previous tasks.