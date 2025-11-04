import torch
import jax
import jax.numpy as jnp
import jax.random as jrandom
import equinox as eqx
import jax.tree_util as jtu
from eqxvision.models.classification.resnet import resnet50
from eqxvision.utils import CLASSIFICATION_URLS
from datasets import load_dataset
from torchvision.transforms.v2 import Compose, Resize, CenterCrop, ToImage, ToDtype, Normalize
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import Dict, Tuple

from jax import config
config.update("jax_default_matmul_precision", "highest")  # disables TF32 usage

from eqxvision.norm_utils import replace_norm

def create_batch_preprocessing_pipeline():
    """Create ImageNet preprocessing pipeline for batch processing."""
    return Compose([
        CenterCrop(224),  # Works on batched tensors: (B, C, H, W)
        Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet mean
            std=[0.229, 0.224, 0.225]    # ImageNet std
        )
    ])

def preprocess_batch(batch_images) -> jnp.ndarray:
    """Convert batch of PIL images to JAX array with proper preprocessing."""
    # Convert PIL images to tensors individually using new v2 API
    to_tensor_transform = Compose([ToImage(), ToDtype(torch.float32, scale=True)])
    tensor_list = []

    for img in batch_images:
        # Ensure image is RGB (3 channels)
        if img.mode != 'RGB':
            img = img.convert('RGB')

        tensor = to_tensor_transform(img)

        # Double-check tensor has 3 channels
        if tensor.shape[0] != 3:
            # If grayscale (1 channel), repeat to get 3 channels
            if tensor.shape[0] == 1:
                tensor = tensor.repeat(3, 1, 1)
            else:
                # Fallback: convert to RGB and retry
                img_rgb = img.convert('RGB')
                tensor = to_tensor_transform(img_rgb)

        tensor_list.append(tensor)

    # Stack into batch tensor: (batch_size, 3, 256, 256)
    tensor_batch = torch.stack(tensor_list)

    # Apply batch transforms (CenterCrop + Normalize work on batched tensors)
    batch_transform = create_batch_preprocessing_pipeline()
    transformed_batch = batch_transform(tensor_batch)  # (batch_size, 3, 224, 224)

    # Convert to JAX array
    return jnp.array(transformed_batch.numpy())

def compute_accuracy(predictions: jnp.ndarray, labels: jnp.ndarray) -> Tuple[float, float]:
    """Compute top-1 and top-5 accuracy."""
    # Get top-5 predictions
    top5_preds = jnp.argsort(predictions, axis=1)[:, -5:]  # Last 5 are highest
    top1_preds = top5_preds[:, -1:]  # Last 1 is highest

    # Compute accuracies
    top1_correct = jnp.any(top1_preds == labels.reshape(-1, 1), axis=1)
    top5_correct = jnp.any(top5_preds == labels.reshape(-1, 1), axis=1)

    top1_acc = jnp.mean(top1_correct).item()
    top5_acc = jnp.mean(top5_correct).item()

    return top1_acc, top5_acc

def count_norms(model):
    """Count different types of normalization layers."""
    from eqxvision.norm_utils import StatelessBatchNorm
    
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

def evaluate_model(model, dataset, num_samples: int = None, batch_size: int = 32) -> Dict[str, float]:
    """
    Evaluate the model on ImageNet validation set.
    
    Args:
        model: The ResNet model
        dataset: Hugging Face dataset
        num_samples: Number of samples to evaluate (None for all)
        batch_size: Batch size for evaluation

    Returns:
        Dictionary containing accuracy metrics
    """
    print("🚀 Starting ImageNet evaluation...")

    # Initialize metrics
    total_samples = 0
    correct_top1 = 0
    correct_top5 = 0

    # Create random key for model inference
    key = jrandom.PRNGKey(42)
    
    # Create vectorized model function for batch processing (done once, outside loop)
    # Note: ResNet expects key as keyword-only argument
    def single_inference(x, key):
        return model(x, key=key)

    vmapped_model = eqx.filter_vmap(single_inference, in_axes=(0, 0))

    # Create batched dataset
    if num_samples:
        dataset = dataset.take(num_samples)

    batched_dataset = dataset.batch(batch_size)

    try:
        with tqdm(desc="Evaluating", unit="batches") as pbar:
            for batch in batched_dataset:
                # Extract batch data
                batch_images = batch['image_pil']
                batch_labels = jnp.array(batch['cls'])

                # Preprocess batch of images
                batch_data = preprocess_batch(batch_images)  # (batch_size, 3, 224, 224)

                # Get predictions using pre-created vmapped model
                key, subkey = jrandom.split(key)
                batch_keys = jrandom.split(subkey, batch_data.shape[0])
                predictions = vmapped_model(batch_data, batch_keys)  # (batch_size, num_classes)

                # Compute batch accuracy
                batch_top1, batch_top5 = compute_accuracy(predictions, batch_labels)

                # Update running metrics
                batch_len = len(batch_labels)
                correct_top1 += batch_top1 * batch_len
                correct_top5 += batch_top5 * batch_len
                total_samples += batch_len

                # Update progress
                pbar.update(1)  # 1 batch processed
                pbar.set_postfix({
                    'Samples': total_samples,
                    'Top-1': f'{correct_top1/total_samples:.3f}',
                    'Top-5': f'{correct_top5/total_samples:.3f}'
                })

    except KeyboardInterrupt:
        print("\n⚠️  Evaluation interrupted by user")

    # Calculate final metrics
    if total_samples > 0:
        final_top1 = correct_top1 / total_samples
        final_top5 = correct_top5 / total_samples

        results = {
            'top1_accuracy': final_top1,
            'top5_accuracy': final_top5,
            'total_samples': total_samples
        }

        print(f"\n📊 Evaluation Results:")
        print(f"   Total samples: {total_samples}")
        print(f"   Top-1 accuracy: {final_top1:.4f} ({final_top1*100:.2f}%)")
        print(f"   Top-5 accuracy: {final_top5:.4f} ({final_top5*100:.2f}%)")

        return results
    else:
        print("❌ No samples processed!")
        return {}

if __name__ == "__main__":
    print("=" * 80)
    print("Evaluating ResNet-50 with BatchNorm → GroupNorm Replacement")
    print("=" * 80)
    
    # Load model with pretrained weights
    print("\n📥 Loading ResNet-50 with PyTorch pretrained weights...")
    model = resnet50(torch_weights=CLASSIFICATION_URLS["resnet50"])
    print("✅ ResNet-50 model loaded successfully with PyTorch weights!")
    
    # Show initial norm layer counts
    print("\n📊 Initial normalization layers:")
    bn_count, gn_count, sbn_count = count_norms(model)
    print(f"   BatchNorm layers: {bn_count}")
    print(f"   GroupNorm layers: {gn_count}")
    print(f"   StatelessBatchNorm layers: {sbn_count}")
    
    # Replace BatchNorm with GroupNorm
    print("\n🔄 Converting BatchNorm layers to GroupNorm...")
    model = replace_norm(model, target="groupnorm")
    print("✅ BatchNorm → GroupNorm conversion completed")
    
    # Show final norm layer counts
    print("\n📊 Final normalization layers:")
    bn_count, gn_count, sbn_count = count_norms(model)
    print(f"   BatchNorm layers: {bn_count}")
    print(f"   GroupNorm layers: {gn_count}")
    print(f"   StatelessBatchNorm layers: {sbn_count}")
    
    if gn_count == 0:
        print("❌ Warning: No GroupNorm layers found after conversion!")
    else:
        print(f"✅ Successfully converted to {gn_count} GroupNorm layers")

    # Load dataset
    print("\n📥 Loading ImageNet validation dataset...")
    dataset = load_dataset("pshishodia/imagenet-1k-256", split="validation", streaming=True)

    # Full evaluation on entire validation dataset
    print("\n🚀 Running full evaluation on ImageNet validation dataset...")
    print("   This will evaluate all ~50,000 samples and may take some time...")
    print("   ⚠️  Note: The model has NOT been fine-tuned after norm replacement!")
    
    # Use larger batch size for better performance on full dataset
    full_results = evaluate_model(model, dataset, num_samples=None, batch_size=256)
    
    if full_results:
        print(f"\n🎯 Final Results Summary:")
        print(f"   Dataset: ImageNet-1K validation")
        print(f"   Model: ResNet-50 (PyTorch weights → JAX/Equinox → GroupNorm)")
        print(f"   Total samples evaluated: {full_results['total_samples']:,}")
        print(f"   Top-1 accuracy: {full_results['top1_accuracy']:.4f} ({full_results['top1_accuracy']*100:.2f}%)")
        print(f"   Top-5 accuracy: {full_results['top5_accuracy']:.4f} ({full_results['top5_accuracy']*100:.2f}%)")
        
        # Compare with expected PyTorch performance
        expected_top1 = 0.7598  # Expected ResNet-50 performance
        diff = full_results['top1_accuracy'] - expected_top1
        print(f"\n📈 Performance Comparison:")
        print(f"   Expected PyTorch ResNet-50 (BatchNorm): ~75.98%")
        print(f"   JAX/Equinox ResNet-50 (GroupNorm): {full_results['top1_accuracy']*100:.2f}%")
        print(f"   Difference: {diff*100:+.2f}%")
        
        if abs(diff) < 0.005:
            print("   ✅ Performance maintained! GroupNorm replacement had minimal impact.")
        elif diff < 0:
            print(f"   ⚠️  Performance degradation of {abs(diff)*100:.2f}%")
            print("      This is expected without fine-tuning after norm layer replacement.")
        else:
            print("   🎉 Performance improved!")
        
        print("\n" + "=" * 80)
        print("Experiment completed!")
        print("=" * 80)
        print("\n💡 Key insights:")
        print("   - BatchNorm uses batch statistics for normalization")
        print("   - GroupNorm uses group-based normalization (independent of batch)")
        print("   - Pretrained BatchNorm weights may not transfer perfectly to GroupNorm")
        print("   - Fine-tuning would likely improve GroupNorm performance")
    else:
        print("❌ Evaluation failed!")

