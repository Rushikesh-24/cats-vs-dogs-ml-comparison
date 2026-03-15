import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
print("GPUs available:", gpus)

import torch

# Check for Apple's Metal Performance Shaders (MPS)
if torch.backends.mps.is_available():
    print("✅ SUCCESS! Mac GPU is ready to go.")
    # Let's create a dummy tensor and send it to the GPU just to be sure
    x = torch.ones(1, device="mps")
    print(f"Test tensor created on: {x.device}")
else:
    print("❌ GPU not found. Still on CPU.")