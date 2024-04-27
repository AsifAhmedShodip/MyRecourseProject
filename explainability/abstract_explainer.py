class AbstractExplainer:
    def __init__(self, model, data):
        self.model = model
        self.data = data

    def generate_explanation(explainer, query_instance, total_CFs):
        counterfactuals = explainer.generate_counterfactuals(query_instances=query_instance,
                                                            total_CFs=total_CFs,
                                                            desired_class="opposite")
        # Check if counterfactuals are generated and not empty
        if counterfactuals is not None and not counterfactuals.cf_examples_list:
            print("No counterfactuals were generated.")
        else:
            # Display all data including unchanged values
            cf_df = counterfactuals.visualize_as_dataframe(show_only_changes=False)
            print("Counterfactuals Generated:\n", cf_df)
            return cf_df