"""Image decoding and resize utility for the image-category model."""

import io
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

IMG_SIZE = (224, 224)


def preprocess_image(image_input):
    """Return a 224x224 RGB float32 batch and the oriented display image."""
    if isinstance(image_input, Image.Image):
        source = image_input.copy()
        encoded = None
    elif isinstance(image_input, (str, Path)):
        encoded = Path(image_input).read_bytes()
        source = Image.open(io.BytesIO(encoded))
    else:
        if hasattr(image_input, "getvalue"):
            encoded = image_input.getvalue()
        elif hasattr(image_input, "read"):
            position = image_input.tell() if hasattr(image_input, "tell") else None
            encoded = image_input.read()
            if position is not None and hasattr(image_input, "seek"):
                image_input.seek(position)
        else:
            encoded = bytes(image_input)
        source = Image.open(io.BytesIO(encoded))

    orientation = source.getexif().get(274, 1)
    image = ImageOps.exif_transpose(source).convert("RGB")
    source.close()
    if encoded is not None and orientation == 1:
        # Match the training decoder and resize operations exactly.
        decoded = tf.io.decode_image(encoded, channels=3, expand_animations=False)
        decoded.set_shape([None, None, 3])
    else:
        # PIL applies EXIF orientation for display and for the rare oriented upload.
        decoded = tf.convert_to_tensor(np.asarray(image, dtype=np.uint8))
    resized = tf.image.resize(decoded, IMG_SIZE, method="bilinear")
    img_array = resized.numpy().astype(np.float32, copy=False)[None, ...]
    return img_array, image
