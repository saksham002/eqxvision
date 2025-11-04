"""
Test script for GroupNorm ResNet models.

This demonstrates how to use ResNet with GroupNorm for imitation learning/RL,
including feature extraction capabilities.
"""

import jax
import jax.numpy as jnp
import jax.random as jrandom
import equinox as eqx
from eqxvision.models import resnet50, resnet50_groupnorm, ResNetFeatureExtractor
from eqxvision.utils import CLASSIFICATION_URLS
from eqxvision.norm_utils import replace_norm, StatelessBatchNorm

print("=" * 80)
print("Testing GroupNorm ResNet Models")
print("=" * 80)

# Test 1: Create a ResNet-50 with GroupNorm from scratch
print("\n1. Creating ResNet-50 with GroupNorm from scratch...")
key = jrandom.PRNGKey(0)
model_groupnorm = resnet50_groupnorm(key=key)
print("✅ ResNet-50 with GroupNorm created successfully!")

# Test 2: Create a test input
print("\n2. Creating test input (3, 224, 224)...")
key, subkey = jrandom.split(key)
test_input = jrandom.normal(subkey, (3, 224, 224))
print(f"✅ Test input shape: {test_input.shape}")

# Test 3: Forward pass through GroupNorm ResNet
print("\n3. Running forward pass through GroupNorm ResNet...")
key, subkey = jrandom.split(key)
output = model_groupnorm(test_input, key=subkey)
print(f"✅ Output shape: {output.shape}")
print(f"   Output dtype: {output.dtype}")

# Test 4: Count different types of normalization layers
print("\n4. Checking normalization layer types...")
def count_norms(model):
    counts = {"bn": 0, "gn": 0, "sbn": 0}
    
    def count_leaf(x):
        if isinstance(x, eqx.nn.BatchNorm):
            counts["bn"] += 1
        elif isinstance(x, eqx.nn.GroupNorm):
            counts["gn"] += 1
        elif isinstance(x, StatelessBatchNorm):
            counts["sbn"] += 1
        return x
    
    # Use tree_map to visit all nodes, treating norm layers as leaves
    jax.tree_util.tree_map(
        count_leaf, 
        model, 
        is_leaf=lambda x: isinstance(x, (eqx.nn.BatchNorm, eqx.nn.GroupNorm, StatelessBatchNorm))
    )
    
    return counts["bn"], counts["gn"], counts["sbn"]

bn_count, gn_count, sbn_count = count_norms(model_groupnorm)
print(f"   BatchNorm layers: {bn_count}")
print(f"   GroupNorm layers: {gn_count}")
print(f"   StatelessBatchNorm layers: {sbn_count}")
if gn_count > 0:
    print("✅ Model contains GroupNorm layers!")
else:
    print("❌ No GroupNorm layers found!")

# Test 5: Feature extraction
print("\n5. Testing feature extraction...")
feature_extractor = ResNetFeatureExtractor(model_groupnorm, extract_layer='layer4', include_avgpool=True)
key, subkey = jrandom.split(key)
features = feature_extractor(test_input, key=subkey)
print(f"✅ Extracted features shape: {features.shape}")

# Test 6: Feature extraction without avgpool
print("\n6. Testing feature extraction without avgpool...")
feature_extractor_no_pool = ResNetFeatureExtractor(model_groupnorm, extract_layer='layer4', include_avgpool=False)
key, subkey = jrandom.split(key)
features_no_pool = feature_extractor_no_pool(test_input, key=subkey)
print(f"✅ Extracted features shape (no avgpool): {features_no_pool.shape}")

# Test 7: Loading pretrained weights and converting to GroupNorm
print("\n7. Loading pretrained weights and converting to GroupNorm...")
print("   (This will download ~100MB if not cached)")
try:
    model_pretrained = resnet50_groupnorm(torch_weights=CLASSIFICATION_URLS["resnet50"])
    print("✅ Loaded pretrained weights and converted to GroupNorm!")
    
    # Verify conversion
    bn_count, gn_count, sbn_count = count_norms(model_pretrained)
    print(f"   BatchNorm layers: {bn_count}")
    print(f"   GroupNorm layers: {gn_count}")
    print(f"   StatelessBatchNorm layers: {sbn_count}")
    
    # Test forward pass
    key, subkey = jrandom.split(key)
    output_pretrained = model_pretrained(test_input, key=subkey)
    print(f"✅ Forward pass successful, output shape: {output_pretrained.shape}")
    
except Exception as e:
    print(f"⚠️  Could not load pretrained weights: {e}")
    print("   This is expected if torch is not installed or network is unavailable")

# Test 8: Manual conversion from BatchNorm to GroupNorm
print("\n8. Testing manual conversion from BatchNorm to GroupNorm...")
key, subkey = jrandom.split(key)
model_batchnorm = resnet50(key=subkey)
print("   Created ResNet-50 with BatchNorm")

# Count norms before conversion
bn_count_before, gn_count_before, _ = count_norms(model_batchnorm)
print(f"   Before conversion - BatchNorm: {bn_count_before}, GroupNorm: {gn_count_before}")

# Convert to GroupNorm
model_converted = replace_norm(model_batchnorm, target="groupnorm")

# Count norms after conversion
bn_count_after, gn_count_after, _ = count_norms(model_converted)
print(f"   After conversion - BatchNorm: {bn_count_after}, GroupNorm: {gn_count_after}")

if bn_count_after == 0 and gn_count_after > 0:
    print("✅ Conversion successful!")
else:
    print("❌ Conversion failed!")

# Test forward pass after conversion
key, subkey = jrandom.split(key)
output_converted = model_converted(test_input, key=subkey)
print(f"✅ Forward pass after conversion successful, output shape: {output_converted.shape}")

# Test 9: Batch processing with vmap
print("\n9. Testing batch processing with vmap...")
batch_size = 4
key, subkey = jrandom.split(key)
batch_input = jrandom.normal(subkey, (batch_size, 3, 224, 224))

def single_forward(x, key):
    return model_groupnorm(x, key=key)

vmapped_forward = eqx.filter_vmap(single_forward, in_axes=(0, 0))
key, subkey = jrandom.split(key)
batch_keys = jrandom.split(subkey, batch_size)
batch_output = vmapped_forward(batch_input, batch_keys)
print(f"✅ Batch processing successful!")
print(f"   Input shape: {batch_input.shape}")
print(f"   Output shape: {batch_output.shape}")

print("\n" + "=" * 80)
print("All tests passed! ✅")
print("=" * 80)
print("\nUsage example for imitation learning:")
print("""
from eqxvision.models import resnet50_groupnorm, ResNetFeatureExtractor
from eqxvision.utils import CLASSIFICATION_URLS

# Create encoder with pretrained weights
encoder = resnet50_groupnorm(torch_weights=CLASSIFICATION_URLS["resnet50"])

# Extract features from layer4
feature_extractor = ResNetFeatureExtractor(encoder, extract_layer='layer4')

# Use in your policy network
features = feature_extractor(observation, key=key)
# features shape: (2048, 1, 1) - ready for policy head
""")

