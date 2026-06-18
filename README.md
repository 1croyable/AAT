[中文版本](Docs/README-CN.md)

# AATField

**Anchor-based Additive Transport Field**

A nonlinear classification model. It treats each sample as a point in a state space and learns an anchor-induced transport field, allowing sample states to move layer by layer within this learnable field and eventually complete classification.

Simply put: *sample state → observe anchors and produce transport direction and strength → residual update → linear readout*

<p align="center">
  <img src="Docs/Imgs/2d_classification.gif" alt="2D classification demo" width="45%" />
  <img src="Docs/Imgs/3d_traces.gif" alt="3D transport traces" width="45%" />
</p>
<p align="center">  <em>Visualization of sample transport in AATField on toy classification tasks.</em></p>

AATField does not aim to stack larger matrices, but instead explores a more geometric and field-like way of evolving representations: **gradually transporting samples in the state space to positions that are easier to classify.**

## Why AATField?

In modern neural networks, many representation learning processes rely on large-scale matrix transformations. Matrix computation is extremely efficient on GPUs and has been proven to be highly powerful.

However, this does not mean that matrix transformation is the only way to evolve representations. AATField comes from a simple geometric intuition:

> If the goal of classification is ultimately to make samples from different classes easier to separate in space, can we directly learn “how to move these samples,” instead of only changing them indirectly through layer-by-layer matrix mappings? This leads to the exploration of a structure whose main source of capacity is not large-scale trainable weight matrices.

This gives the model a very intuitive interpretation: **each layer applies a set of local geometric transports to the sample state, gradually moving samples into a more separable spatial structure**.

## Quick Start

### 1. Installation

It is recommended to install dependencies in a virtual environment:

```bash
pip install git+https://github.com/1croyable/AATField.git@main
```

If the project has been configured for local package installation, you can also use:

```bash
pip install -e .
```

### 2. Basic Usage

```python
import torch
import torch.nn.functional as F

from aatfield import AATField, AATFieldConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cfg = AATFieldConfig.from_data(
    train_x,
    num_classes=xxx, # number of classes for this task
    extra_dims=xxx, # number of additional dimensions
    layers=xxx, # number of layers
    max_children=xxx, # maximum number of child anchors inferred during initialization for each layer
)

model = AATField(cfg).to(device)

# The model needs to be structurally initialized from the training data first
model.initialize(train_x, train_y)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3) # create the optimizer after initialization

for x, y in train_loader:
    x = x.to(device)
    y = y.to(device)

    logits = model(x)
    loss = F.cross_entropy(logits, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### 3. Inference

```python
model.eval()
with torch.no_grad():
    logits = model(x)
    pred = logits.argmax(dim=1)
```

## Model Structure

![structure](Docs/Imgs/structure.jpg)

AATField can be summarized as three actions: observe, transport, classify.

### 1. State Space: Observing Samples in a Higher-Dimensional Space

AATField first lifts the original input into a higher-dimensional state space.

The model then places a set of learnable anchors in the state space. These anchors are not ordinary hidden neurons, but reference points used to define local geometric fields. Each sample obtains different response strengths according to its distance relationships with the anchors.

This design comes from an intuitive assumption: the visible data we have may only be a low-dimensional projection of higher-dimensional real information. A classification result may be affected by many unobserved factors, and the model cannot directly recover these hidden factors. Therefore, AATField does not try to reconstruct the true high-dimensional information. Instead, it **learns a transport field in the expanded state space to reorganize the existing projected information**, making samples gradually easier to classify.

------

### 2. Generating Transport: Direction, Strength, and Activation

Each anchor produces a set of contributions for a sample, and multiple contributions are summed into an overall transport vector. A contribution contains:

- Direction: where the sample should move;
- Strength: how large this movement should be;

Multiple anchor contributions are combined into an overall transport vector and then used to update the sample state in a residual form. To enhance nonlinear expressiveness, AATField can introduce nonlinear modulation during transport generation or after state update, so that the transport field does not merely bend space smoothly, but can also form more flexible local deformations, boundary turns, and state rearrangements.

------

### 3. Linear Readout: Making the Transported Space Separable

AATField finally uses a simple linear classification head. This design gives the transport process a clear objective: after multiple layers of transport, samples should become as linearly separable as possible in the final state space.

It does not compress all points into a fixed shape. Instead, it **locally stretches, folds, and rearranges the state space like twisting clay**, making the classification boundary simpler in the final space.

## Current Experimental Observations

Preliminary experiments show that:

- AATField can form effective nonlinear boundaries on two-dimensional geometric classification tasks;
- On low-dimensional geometric tasks such as moons, spiral, and checkerboard, AATField shows a strong geometric inductive bias;
- In some small-parameter settings, AATField can approach or outperform MLP baselines with similar parameter counts;
- On flat image and tabular tasks, it has not yet stably surpassed strong MLP/CNN baselines;
- The transport process has good visual interpretability, allowing direct observation of how sample points move layer by layer in the state space.

## License

This project uses the MIT License.

You are free to use, modify, and distribute the project code, but the original copyright notice must be retained.