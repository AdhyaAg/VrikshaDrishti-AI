"""Disabled legacy diagnostic recommendations.

The available model predicts image framing categories only. It has no validated
species, disease, or plant-health targets, so treatment recommendations cannot
be made from its output.
"""


def get_recommendation(class_name):
    """Return neutral scope text instead of an unsupported diagnosis/treatment."""
    return {
        "species": "Not identified",
        "status": "Image category only",
        "badge_color": "#2E7D32",
        "diagnosis": f"Image category: {class_name}. Species and plant health are not classified.",
        "symptoms": "Disease symptoms are not classified by this model.",
        "organic_remedy": "No treatment recommendation is supported by an image-category result.",
        "chemical_remedy": "Do not apply chemical treatment based on this model output.",
        "pune_eco_tip": "For health concerns, seek a qualified plant-health review.",
    }
