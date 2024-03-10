# Meta Continual Learning

## Introduction

Meta Continual Learning (MCL) represents a cutting-edge strategy designed to navigate the challenges of catastrophic forgetting in neural networks, which traditionally occurs when models are exposed to new tasks sequentially. This approach leverages the principles of meta-learning, or "learning to learn," enabling a model to swiftly adapt to new information without discarding valuable knowledge from earlier tasks. MCL achieves this by optimizing the model's learning process itself, ensuring that it becomes more efficient and effective at acquiring new skills over time, while simultaneously preserving past learnings. This method not only enhances the model's ability to handle a variety of tasks in a continual learning scenario but also significantly improves its generalization capabilities across tasks, embodying a dynamic balance between retaining old knowledge and embracing new experiences.

## Implementation

### Data Preprocessing

In my first step, I divided the FashionMNIST dataset into distinct tasks to facilitate task-specific training and evaluation:

```Python
# A function to split the FashionMNIST dataset into tasks
def create_fashion_mnist_tasks(root_dir, num_tasks=5, transform=None):
    # Load the entire FashionMNIST dataset
    full_dataset = FashionMNIST(root=root_dir, train=False, download=True, transform=transform)

    # Determine the number of classes per task
    classes_per_task = len(full_dataset.classes) // num_tasks

    # Create tasks by splitting the dataset
    tasks = []
    for task_idx in range(num_tasks):
        # Calculate class indices for the current task
        class_start = task_idx * classes_per_task
        class_end = class_start + classes_per_task
        task_classes = list(range(class_start, class_end))

        # Find indices of images belonging to the current task's classes
        task_indices = [i for i, (_, label) in enumerate(full_dataset) if label in task_classes]

        # Create a Subset for the current task
        task_dataset = Subset(full_dataset, task_indices)
        tasks.append(task_dataset)

    return tasks

# Create tasks
root_dir = './data'
tasks_evaluation = create_fashion_mnist_tasks(root_dir, num_tasks=5, transform=transform)
```

I aimed to systematically split the dataset, ensuring each task had a distinct set of classes, which was crucial for evaluating the model's ability to learn and adapt across tasks.

### Neural Network Architecture

Next, I designed a simple CNN architecture suitable for the task at hand:

```Python
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = nn.functional.relu(nn.functional.max_pool2d(self.conv1(x), 2))
        x = nn.functional.relu(nn.functional.max_pool2d(self.conv2(x), 2))
        x = x.view(x.size(0), -1)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```

My goal was to construct a model that was both straightforward and effective for the classification tasks derived from FashionMNIST.

### Training

I applied a MAML-like update strategy for training, focusing on quickly adapting to new tasks:

```Python
def maml_update(model, optimizer, loss_fn, data_loader, steps=3, alpha=0.0001):
    # Create a copy of the model's initial state
    initial_state = {name: param.clone() for name, param in model.named_parameters()}

    # Task-specific update
    for step in range(steps):
        for inputs, labels in data_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            # labels = torch.tensor(labels)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            # Apply the update to the model (simulating gradient descent)
            with torch.no_grad():
                for name, param in model.named_parameters():
                    param -= alpha * param.grad

    # Update the meta-model parameters
    meta_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    meta_loss = 0
    for inputs, labels in data_loader:
        outputs = model(inputs)
        meta_loss += loss_fn(outputs, labels)

    meta_optimizer.zero_grad()
    meta_loss.backward()
    meta_optimizer.step()

    # Restore the model to its initial state
    with torch.no_grad():
        for name, param in model.named_parameters():
            param.copy_(initial_state[name])
```

I devised this function to encapsulate the essence of MCL, ensuring my model could swiftly adapt through task-specific updates while also undergoing a meta-update to refine its parameters across all tasks.

```Python
for task_id, task_loader in enumerate(task_dataloaders):
    print(f"Training on task {task_id}")
    maml_update(model, optimizer, nn.CrossEntropyLoss(), task_loader)
```
This loop was critical for applying my MAML update function across each task, sequentially training my model to handle a variety of challenges.

### Evaluation and Adjustments

Despite implementing MCL strategies to combat catastrophic forgetting, the accuracies observed—30 and 31% for the first two tasks, with the others at zero—underscored the complexity of balancing new task acquisition against retaining knowledge from previous tasks.

### Adjustment Strategies

* __Fine-tuning Learning Rates__: I experimented with learning rates for both the task-specific and meta-updates, seeking an optimal balance that enhanced knowledge retention.
* __Enhancing Task Diversity__: I considered introducing a wider array of tasks during training to promote more generalized learning.
* __Optimizing Task Division__: Adjusting how tasks were divided, perhaps ensuring a more strategic class distribution, could potentially aid in better overall retention.
* __Exploring More Complex Models__: A deeper or more intricate network architecture might better capture features, improving task generalization.
* __Incorporating Regularization__: Techniques like dropout or weight decay might help prevent overfitting to a specific task, preserving earlier task performance.