# eqxvision Examples

This directory contains example scripts demonstrating how to use eqxvision models.

## Quick Start

### 1. Basic Usage: ImageNet Pretrained ResNet

Load a ResNet-50 with pretrained ImageNet weights and run inference:

```python
import jax.random as jrandom
from eqxvision.models import resnet50
from eqxvision.utils import CLASSIFICATION_URLS

# Load pretrained model
model = resnet50(torch_weights=CLASSIFICATION_URLS["resnet50"])

# Run inference
key = jrandom.PRNGKey(0)
image = jrandom.normal(key, (3, 224, 224))  # (C, H, W)
logits = model(image, key=key)  # (1000,) ImageNet classes
```

### 2. ResNet with GroupNorm (for RL/Imitation Learning)

For environments with small batch sizes or single-sample inference:

```python
from eqxvision.models import resnet50_groupnorm
from eqxvision.utils import CLASSIFICATION_URLS

# Option A: Create from scratch with GroupNorm
model = resnet50_groupnorm(key=key)

# Option B: Load pretrained weights, then convert to GroupNorm
model = resnet50_groupnorm(torch_weights=CLASSIFICATION_URLS["resnet50"])
```

### 3. Manual BatchNorm → GroupNorm Conversion

Convert any model's BatchNorm layers to GroupNorm:

```python
from eqxvision.models import resnet50
from eqxvision.norm_utils import replace_norm
from eqxvision.utils import CLASSIFICATION_URLS

# Load pretrained model with BatchNorm
model = resnet50(torch_weights=CLASSIFICATION_URLS["resnet50"])

# Convert all BatchNorm layers to GroupNorm
model = replace_norm(model, target="groupnorm")

# Now ready for small-batch or single-sample inference
```

### 4. Feature Extraction

Extract intermediate features for downstream tasks:

```python
from eqxvision.models import ResNetFeatureExtractor, resnet50_groupnorm
from eqxvision.utils import CLASSIFICATION_URLS

# Create encoder
encoder = resnet50_groupnorm(torch_weights=CLASSIFICATION_URLS["resnet50"])

# Extract features from layer4 (before final FC layer)
feature_extractor = ResNetFeatureExtractor(
    encoder, 
    extract_layer='layer4',
    include_avgpool=True  # Output: (2048, 1, 1)
)

# Use in your policy/value network
features = feature_extractor(observation, key=key)
```

### 5. Batch Processing with vmap

Efficient batch inference:

```python
import equinox as eqx
import jax.random as jrandom

# Create batched forward function
def single_forward(x, key):
    return model(x, key=key)

vmapped_forward = eqx.filter_vmap(single_forward, in_axes=(0, 0))

# Process batch
batch_size = 32
batch_images = jrandom.normal(key, (batch_size, 3, 224, 224))
batch_keys = jrandom.split(key, batch_size)
batch_logits = vmapped_forward(batch_images, batch_keys)  # (32, 1000)
```

## Available Scripts

### `eval_resnet.py`
Evaluates a pretrained ResNet-50 on ImageNet validation set (~50k images).

```bash
python examples/eval_resnet.py
```

Expected performance: **~75.98% top-1 accuracy**

### `eval_resnet_groupnorm.py`
Evaluates the effect of replacing BatchNorm with GroupNorm on a pretrained model (without fine-tuning).

```bash
python examples/eval_resnet_groupnorm.py
```

This experiment demonstrates:
- How to convert pretrained models from BatchNorm to GroupNorm
- Performance impact of norm layer replacement without fine-tuning
- Typical performance drop: 2-5% (varies by model and dataset)

### `test_groupnorm_resnet.py`
Comprehensive test suite for GroupNorm ResNet models, including:
- Creating models with GroupNorm from scratch
- Loading pretrained weights and converting to GroupNorm
- Feature extraction
- Batch processing with vmap

```bash
python examples/test_groupnorm_resnet.py
```

## When to Use GroupNorm vs BatchNorm

| Use Case | Recommended Norm | Reason |
|----------|------------------|--------|
| Standard image classification | BatchNorm | Better performance with batch training |
| Reinforcement Learning | GroupNorm | Works with batch_size=1 |
| Imitation Learning | GroupNorm | Independent of batch statistics |
| Online inference (single images) | GroupNorm or StatelessBatchNorm | No batch dependency |
| Fine-tuning with small batches | GroupNorm | Stable with small batch sizes |

## Stateless BatchNorm for Inference

For inference-only use cases, convert BatchNorm to a stateless version:

```python
from eqxvision.norm_utils import replace_norm

# Convert to stateless (uses fixed running statistics)
model = replace_norm(model, target="stateless")

# No state tracking needed, no vmap axis_name required
output = model(image, key=key)
```

## Notes

- **Image format**: eqxvision expects `(C, H, W)` format (channels first)
- **Preprocessing**: Apply ImageNet normalization for pretrained models:
  - Mean: `[0.485, 0.456, 0.406]`
  - Std: `[0.229, 0.224, 0.225]`
- **Key argument**: All models require a PRNG key for dropout/stochastic operations (even if unused)

## Available Pretrained Models

```python
from eqxvision.utils import CLASSIFICATION_URLS

# ResNet family
CLASSIFICATION_URLS["resnet18"]
CLASSIFICATION_URLS["resnet34"]
CLASSIFICATION_URLS["resnet50"]
CLASSIFICATION_URLS["resnet101"]
CLASSIFICATION_URLS["resnet152"]
```

All weights are automatically downloaded from PyTorch and converted to JAX/Equinox format.

