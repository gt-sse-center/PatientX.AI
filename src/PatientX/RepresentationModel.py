from enum import Enum

class RepresentationModel(str, Enum):
    """
    Enum for representation algorithm options
    """
    mistral_small = "mistral-small",
    gpt4o = "gpt4o"
