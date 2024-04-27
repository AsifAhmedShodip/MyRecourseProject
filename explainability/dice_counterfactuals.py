# dice_counterfactuals.py
from explainability.abstract_explainer import AbstractExplainer
import dice_ml
from dice_ml import Data, Model, Dice

class DiceCounterfactuals(AbstractExplainer):
    def __init__(self, model, data, continuous_features, outcome_name):
        super().__init__(model, data)
        self.dice_data = Data(dataframe=self.data,
                              continuous_features=continuous_features,
                              outcome_name=outcome_name)
        self.dice_model = Model(model=self.model, backend='sklearn')
        self.explainer = Dice(self.dice_data, self.dice_model)

    def generate_explanation(self, query_instance, total_CFs=3, desired_class="opposite"):
        counterfactuals = self.explainer.generate_counterfactuals(query_instances=query_instance,
                                                                  total_CFs=total_CFs,
                                                                  desired_class=desired_class)
        return counterfactuals.visualize_as_dataframe(show_only_changes=True)
