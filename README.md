[中文版本](Docs/README-CN.md)

# Anchor-based Additive Transport (AAT)

AAT is a nonlinear classification algorithm. It first transforms the input into a normalized radial-directional state, then allows each sample to respond to a set of trainable ray anchors. The resulting response weights are used to mix the transport values carried by each ray, so that samples gradually become easier to classify through multiple layers of additive transport.

In simple terms: *input features → centering and polar transform → ray response and weight computation → additive transport → linear readout*

<p align="center">
  <img src="Docs/Imgs/2d_classification.gif" alt="2D classification demo" width="70%" />
</p>
<p align="center">  <em>Visualization of sample transport in AAT on a 2D checkerboard classification task.</em></p>

AAT does not use stacks of large dense matrices as its primary modeling mechanism. Instead, it updates sample states layer by layer by using state-dependent ray responses to mix trainable transport values, and finally performs classification with a linear head.

## Why AAT?

In modern neural networks, many representation-learning processes rely on large-scale matrix transformations. Matrix computation is highly efficient on GPUs and has already proven to be extremely powerful.

However, this does not mean that matrix transformation is the only way to evolve data representations. AAT originates from a simple geometric intuition:

> If the ultimate goal of classification is to make samples from different classes easier to distinguish in space, can we directly learn “how to move these samples,” rather than changing them only indirectly through layer-by-layer matrix mappings? This motivates the exploration of a structure that does not use large-scale dense matrix transformations as its primary modeling mechanism.

This gives the model a very intuitive interpretation: **each layer computes how strongly the current sample state responds to a set of trainable rays, then uses the response weights to mix the corresponding transport values, gradually producing a state representation that is easier to separate.**

## Quick Start

### 1. Installation

It is recommended to install the dependencies in a virtual environment:

```bash
pip install git+https://github.com/1croyable/AAT.git@main
```

If the project has already been configured for local package installation, you can also use:

```bash
pip install -e .
```

### 2. Basic Usage

```python
import torch
import torch.nn.functional as F

from aat import AAT, AATConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cfg = AATConfig.from_data(
    train_x,
    num_classes=xxx,  # number of classes
    layers=xxx,       # number of transport layers
    rays=xxx,         # number of rays in each layer
)

model = AAT(cfg).to(device)

# Fit the center and radial normalization range from training data
model.fit_state(train_x)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for x, y in train_loader:
    x = x.to(device)
    y = y.to(device)

    logits = model(x)
    loss = F.cross_entropy(logits, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

`rays` can be an integer, meaning that all layers use the same number of rays:

```
rays=32
```

It can also specify a different number of rays for each layer:

```
rays=[16, 24, 32, 32]
```

### 3. Inference

```python
model.eval()

with torch.no_grad():
    x = x.to(device)
    logits = model(x)
    pred = logits.argmax(dim=1)
```

### 4. Save and Load

```python
model.save_checkpoint("aat.pt") # Save both the configuration and trained weights
```

```python
model = AAT.from_checkpoint("aat.pt", map_location=device) # Load the model
```

## Model Structure

![structure](Docs/Imgs/structure.jpg)

AAT can be summarized in three stages: state transformation, additive transport, and linear classification.

### 1. Polar State Construction

AAT first centers the input features and transforms each sample into a polar state composed of a normalized radius $\rho$ and a unit direction vector $\mathbf{u}$:

$$
\mathbf{z}_0=[\rho_0,\mathbf{u}_0]
$$

------

### 2. Ray Response and Additive Transport

Each layer contains a set of trainable rays. Based on its current state, each sample computes a response score for every ray, and softmax is then used to obtain the corresponding response weights.

Each ray also carries an independent pair of radial and directional transport values. The model uses the response weights to form a weighted mixture of these transport values and updates the current state in residual form. The direction vector is renormalized after every update.

This process is repeated across multiple layers, gradually transforming the sample state.

------

### 3. Linear Readout

After $L$ transport layers, the final state is represented as:

$$
\mathbf{z}_L=[\rho_L,\mathbf{u}_L]
$$

AAT uses a simple linear classification head to directly read the final state and produce class predictions. Training therefore encourages the preceding transport layers to transform samples into state representations that are easier to classify linearly.

## Current Experimental Observations

Current experiments show that:

- AAT can learn complex nonlinear classification boundaries on low-dimensional tasks such as checkerboard, Swiss roll, and triple helix, reaching representative test accuracies of approximately 97.8%, above 97%, and around 94%, respectively;
- On real-world tabular datasets such as Airline Satisfaction, AAT demonstrates clearly better parameter efficiency than MLP baselines at the same or similar parameter counts, especially in small-model settings, while also achieving higher classification performance in some configurations.
- On flattened 784-dimensional MNIST, an AAT model with 8 layers and 48 rays per layer achieved a best validation accuracy of 98.51% and a test accuracy of 98.26%;
- These results indicate that AAT is not limited to low-dimensional visualization tasks, and can also scale to tabular data and high-dimensional flattened images;
- AAT still has room for improvement in training speed and hardware execution efficiency;
- The transport process can be visualized layer by layer, making it possible to directly observe how sample states change through multiple transport layers and gradually become easier to classify, as illustrated on MNIST.

<img src="Docs/Imgs\mnist_visual.png" alt="mnist_visual" style="zoom: 25%;" />

## License

This project uses the MIT License.

You are free to use, modify, and distribute the project code, provided that the original copyright notice is retained.
