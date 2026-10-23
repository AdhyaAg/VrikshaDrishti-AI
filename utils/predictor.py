import numpy as np
import tensorflow as tf
from utils.model_loader import load_tree_model, load_class_names, load_model_metadata
from utils.image_processing import preprocess_image
from utils.project_data import display_class_name

def predict_tree_health(image_input):
    """
    Predicts the audited coarse image-content category. This model does not
    identify botanical species or diagnose plant health.
    Returns structured results:
    - top_label: Category name ('leafs', 'saplings', 'trees', 'unclear')
    - confidence: Category confidence percentage (0 - 100)
    - health_score: None (no health model is available)
    - status: category-only status or uncertainty
    - species: friendly image-category description for existing report fields
    - top5: Category confidence probability tuples
    """
    model = load_tree_model()
    class_names = load_class_names()
    metadata = load_model_metadata()

    # Preprocess image
    img_array, pil_img = preprocess_image(image_input)

    # Inference
    raw_preds = model.predict(img_array, verbose=0)[0]

    # Apply softmax if needed
    if np.sum(raw_preds) > 1.05 or np.min(raw_preds) < 0:
        probs = tf.nn.softmax(raw_preds).numpy()
    else:
        probs = raw_preds

    if len(probs) != len(class_names):
        raise ValueError("Loaded class mapping does not match the model output size.")
    if not np.isfinite(probs).all():
        raise ValueError("The model returned a non-finite prediction.")

    top_idx = int(np.argmax(probs))
    confidence = float(probs[top_idx] * 100)
    predicted_class = class_names[top_idx]
    uncertainty_threshold = float(metadata.get("uncertainty_threshold", 0.45))
    top_label = predicted_class if confidence / 100.0 >= uncertainty_threshold else "unclear"

    # Calculate Top probability breakdown
    top5_indices = np.argsort(probs)[::-1][:min(len(class_names), 5)]
    top5_results = []
    for idx in top5_indices:
        label = class_names[idx]
        conf = float(probs[idx] * 100)
        top5_results.append((label, conf))

    if top_label == "unclear":
        final_status = "Category uncertain"
        badge_color = "#FBC02D"
        diagnosis = (
            "The image category is uncertain. Retake a well-lit photo with the leaf or whole plant clearly in frame."
        )
        category_description = "Uncertain image category"
    else:
        final_status = "Image category only"
        badge_color = "#2E7D32"
        category_description = display_class_name(top_label)
        diagnosis = (
            f"Image category: {category_description} ({confidence:.2f}% classifier confidence). "
            "Species and plant health are not classified by this model."
        )

    symptoms = "This model does not classify disease symptoms; visible spots or discoloration need expert review."

    return {
        "top_label": top_label,
        "confidence": round(confidence, 2),
        "health_score": None,
        "status": final_status,
        "badge_color": badge_color,
        "species": category_description,
        "diagnosis": diagnosis,
        "symptoms": symptoms,
        "organic_remedy": "No treatment is recommended because this model does not determine plant health.",
        "chemical_remedy": "Do not apply chemical treatment based on this image-category result.",
        "pune_eco_tip": "For health concerns, capture close-up leaf images and seek a qualified plant-health review.",
        "top5": top5_results,
        "pil_image": pil_img
    }
