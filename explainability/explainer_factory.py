# explainer_manager.py
from explainability.dice_counterfactuals import DiceCounterfactuals

def get_explainer(type, *args, **kwargs):
    if type == 'dice':
        return DiceCounterfactuals(*args, **kwargs)
    else:
        raise ValueError("Unsupported explainer type")
