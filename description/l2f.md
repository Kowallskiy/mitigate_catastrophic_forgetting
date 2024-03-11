# Learning Without Forgetting (L2F)

## Introduction

Learning Without Forgetting (L2F) is a strategy designed to enable neural networks to learn new tasks without losing performance on previously learned tasks, effectively addressing the challenge of catastrophic forgetting. Unlike methods that explicitly regularize the neural network's weights, L2F focuses on preserving the output for old tasks when learning new ones. This is achieved by incorporating knowledge distillation, where the network is trained to produce outputs on new tasks that are consistent with those from the original model, thus maintaining its performance across all tasks. L2F seamlessly integrates the learning of new information with the retention of old, ensuring that the acquisition of new skills does not overwrite valuable, previously learned knowledge.

## Implementation

### Math Behind the Algorithm

#### Distillation Loss

The distillation loss, which helps transfer knowledge from the old task (model) to the new one, can be defined as follows:

```math
\mathcal{L}_{distill} = \sum_{i} T^{2} \cdot KL(\sigma(\frac{z_{i}^{old}}{T}) || \sigma(\frac{z_{i}^{new}}{T}))
```

Here, $KL$ denotes the Kullback-Leibler divergence, $\sigma$ represents the softmax function, $z_{i}^{old}$ are the logits from the original model, $z_{i}^{new}$ are the logits from the updated model, a $T$ is the temperature parameter that controls the softening of probabilities.

#### Cross-Entropy for the New Task

The cross-entropy loss for learning the new task is given by:

```math
\mathcal{L}_{new} = -\sum_{j} y_{j} \log(p_{j})
```

where $y_{j}$ is the true label and $p_{j}$ is the predicted probability for class $j$.

#### Overall Objective

The overall objective that combines the distillation loss with the new task's loss is typically formulated as:

```math
\mathcal{L} = (1 - \alpha) \cdot \mathcal{L}_{new} + \alpha \cdot \mathcal{L}_{distill}
```

where $\alpha$ is a balancing coefficient that controls the importance of the distillation loss relative to the new task's loss.

### Base CNN Architecture

I began by defining a base Convolutional Neural Network (CNN) architecture for the initial task. This network comprised several convolutional layers for feature extraction, followed by fully connected layers for the task of classification.

```Python
class CNN(nn.Module):
    def __init__(self, num_classes=10, hidden_size=512):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1)
        self.relu2 = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32*14*14, hidden_size)
        self.relu3 = nn.ReLU()
        self.classifier = nn.Linear(hidden_size, num_classes)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='sigmoid')

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu3(x)
        logits = self.classifier(x)

        return logits

    def forward_features(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu3(x)

        return x
```

### Training on Task 1

I then trained this model on the first task. This phase involved the standard training steps: forward pass, computing the loss, backward pass, and updating the weights. I achieved a 94% accuracy rate.

### Modified CNN for Task 2

For the second task, I introduced a modified CNN that included an additional linear layer and a new classifier. This modification was designed to accommodate learning the new task while retaining knowledge from the first task.

```Python
class ModifiedCNN(CNN):
    def __init__(self, num_classes, hidden_size, new_hidden_size):
        super(ModifiedCNN, self).__init__(num_classes=num_classes, hidden_size=hidden_size)
        self.new_fc = nn.Linear(hidden_size, new_hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.classifier = nn.Linear(new_hidden_size, num_classes)

    def forward(self, x):
        x = super().forward_features(x)
        x = self.new_fc(x)
        x = self.relu(x)
        x = self.classifier(x)
        return x
```

### Weights Transferring

First, I transferred the weights from the model trained on task 1 to the new model, which now included a new hidden layer and classifier:

```Python
conv1_weight = model_task1.conv1.weight.data
conv1_bias = model_task1.conv1.bias.data
conv2_weight = model_task1.conv2.weight.data
conv2_bias = model_task1.conv2.bias.data
fc1_weight = model_task1.fc1.weight.data
fc1_bias = model_task1.fc1.bias.data

model_task2.conv1.weight = nn.Parameter(conv1_weight)
model_task2.conv1.bias = nn.Parameter(conv1_bias)
model_task2.conv2.weight = nn.Parameter(conv2_weight)
model_task2.conv2.bias = nn.Parameter(conv2_bias)
model_task2.fc1.weight = nn.Parameter(fc1_weight)
model_task2.fc1.bias = nn.Parameter(fc1_bias)
```
This transfer ensured that the second model had 6 trained and 4 untrained (empty) outputs.

```Python
in_features = model_task1.classifier.in_features
out_features = model_task1.classifier.out_features

new_weights = torch.zeros_like(model_task2.classifier.weight)
new_biases = torch.zeros_like(model_task2.classifier.bias)

weight = model_task1.classifier.weight.data
bias = model_task1.classifier.bias.data

new_weights[:out_features, :] = weight
new_biases[:out_features] = bias

model_task2.classifier.weight = nn.Parameter(new_weights)
model_task2.classifier.bias = nn.Parameter(new_biases)
```

Then I froze all the parameters except for the classifier and the new hidden layer:

```Python
for name, param in model_task2.named_parameters():
    if name not in ['classifier.weight', 'classifier.bias', 'new_fc.weight', 'new_fc.bias']:
        param.requires_grad = False
```

### Training on Task 2

For training on task 2, I defined a specialized training function that incorporated both the original task's learned knowledge and the new task's learning objectives through a distillation loss.

```Python
def train(alpha, T):
    size = len(train_dataloader_task2.dataset)
    # We set net_new to evaluation mode to prevent it from being updated
    # while computing the distillation loss from the old model

    model_task2.train()
    for batch, (X, y) in enumerate(train_dataloader_task2):
        X, y = X.to(device), y.to(device)

        outputs = model_task2(X)
        soft_y = model_task1(X)

        loss1 = loss_fn(outputs, y)

        outputs_S = nn.functional.softmax(outputs[:, :out_features] / T, dim=1)
        outputs_T = nn.functional.softmax(soft_y[:, :out_features] / T, dim=1)

        loss2 = outputs_T.mul(-1 * torch.log(outputs_S))
        loss2 = loss2.sum(1)
        loss2 = loss2.mean() * T * T

        loss = loss1 + alpha * loss2

        loss.backward(retain_graph=True)
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch+1) * len(X)
            print(f"Loss: {loss:>7f}, {current:>5d}/{size:>5d}")
```

This approach aimed to balance the model's ability to learn from the new task while retaining the previously acquired knowledge.

```Python
T = 2
alpha = 0.9
loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model_task2.parameters(), lr=0.0001, momentum=0.9, weight_decay=5e-4)

for epoch in range(5):
    print(f"Epoch {epoch+1}: ----------------------")
    if epoch > 2:
        # After 2 epochs unfreezee all the parameters
        for param in model_task2.parameters():
            param.requires_grad = True
        optimizer = torch.optim.SGD(model_task2.parameters(), lr=0.0001, momentum=0.9)
    train(alpha, T)
    test(alpha, T)
    val(epoch)
```

### Evaluation and Adjustments

Upon evaluating the model's performance on the dataset from task 1, I observed a 38% accuracy rate. This outcome highlighted the challenges and intricacies involved in implementing L2F effectively, underscoring the importance of continual learning strategies in the advancement of neural networks.
