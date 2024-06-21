import os, pathlib
from synthetic_data.nodes import *
from synthetic_data.edges import *
from bisect import bisect

class Mirror():
    def __init__(self, seed=0):
        self.seed = seed
        np.random.seed(seed)
        self.df = None
        self.cat_cols = []
        self.num_cols = []

    def generate_csv(self, nodes, edges):
        """
        :param nodes: list of Node object. The order represents the order to generate the nodes.
                      E.g. [CategoricalNode("G", [], [], {"M": 0.5, "F": 0.5}, sample_n=100),
                            CategoricalNode("R", [], [], {"W": 0.5, "B": 0.5}, sample_n=100),
                            OrdinalLocalNode("X", [], [], {"bound": [1, 5, 50], "probability": [0.5, 0.5]}, sample_n=100)]
        :param edges: dict, key is the name of the Node object, value is Edge object that represents the incoming edges and its weight for this node.
                      E.g. {"X": ([CtoN("G", "X"), CtoN("R", "X")], [0.5, 0.5])} for NUM and ORD,
                           {"D": [CtoC("G", "D"), NtoC("A", "D")]} for CAT with multiple parents,
                           {"D": CtoC("G", "D")} for CAT with single parent
        :return:
        """
        df = pd.DataFrame()
        for node_i in nodes:
            if node_i.type == "NUM":
                self.num_cols.append(node_i.name)
            else:
                self.cat_cols.append(node_i.name)
            if node_i.name in edges.keys(): # have parents
                print(node_i.name, "with parents")
                # iterate the incoming edges from its parents
                if type(edges[node_i.name]) not in [tuple, list]:  # only have one parent node
                    if edges[node_i.name].parent_name not in df.columns:
                        print("The parent is not exited!")
                        raise ValueError
                    df[node_i.name] = edges[node_i.name].instantiate_values(df)
                    print("One parent", edges[node_i.name], list(df.columns))
                else:  # have more than one parent node, update the inputs probability table based on its parents
                    if type(edges[node_i.name]) == tuple:
                        parents_i = [x.parent_name for x in edges[node_i.name][0]]
                    else:
                        parents_i = [x.parent_name for x in edges[node_i.name]]
                    if len(set(parents_i).intersection(df.columns)) != len(parents_i):
                        print("Some parents are not exited!")
                        raise ValueError
                    if node_i.type == "CAT": # current node is CAT
                        df["group"] = "" # get all the possible subgroups from all the parents' categories
                        for incoming_edge_i, weight_i in zip(edges[node_i.name][0], edges[node_i.name][1]):
                            # print("---", node_i.name, incoming_edge_i.parent_name, incoming_edge_i.child_name)
                            if incoming_edge_i.type[0] == "N":  # get the categories of the numerical node
                                # print("**", incoming_edge_i.bounds, bisect(incoming_edge_i.bounds, 23))
                                df["C_"+incoming_edge_i.parent_name] = df[incoming_edge_i.parent_name].apply(lambda x: str(bisect(incoming_edge_i.bounds,x)))
                                df["group"] += df["C_"+incoming_edge_i.parent_name]
                            else:
                                df["group"] += df[incoming_edge_i.parent_name]
                        # compute the new probability table for the child node considering all possible subgroups
                        all_cpt = {}
                        for gi in df["group"].unique():
                            gi_probability = {}
                            for node_value_i in node_i.domain:
                                prob_i = 0
                                for incoming_edge_i, weight_i in zip(edges[node_i.name][0], edges[node_i.name][1]):
                                    gi_idx = edges[node_i.name][0].index(incoming_edge_i)
                                    prob_i += weight_i * incoming_edge_i.probability_table["".join(gi)[gi_idx]][node_value_i]
                                # for gi_idx, parent_i in enumerate(edges[node_i.name][0]):
                                #     prob_i *= parent_i.probability_table["".join(gi)[gi_idx]][node_value_i]
                                gi_probability[node_value_i] = prob_i
                            # all_cpt["".join(gi)] = {x: gi_probability[x]/sum(gi_probability.values()) for x in gi_probability}
                            all_cpt["".join(gi)] = {x: gi_probability[x] for x in gi_probability}
                        # print("New CPT", all_cpt, "\n")
                        # sample the value of the child node using above new cpt table
                        df[node_i.name] = df["group"].apply(lambda x: np.random.choice(list(all_cpt[x].keys()), p=list(all_cpt[x].values())))
                        # print("Child node is CAT", list(df.columns))
                    else: # the child node is NUM or ORD
                        df[node_i.name] = 0
                        for incoming_edge_i, weight_i in zip(edges[node_i.name][0], edges[node_i.name][1]):
                            values_i = incoming_edge_i.instantiate_values(df)
                            df[node_i.name] = df[node_i.name] + weight_i * values_i
                        # print("Child node is numerical", list(df.columns))

            else: # no parents
                # instantiate using its parameters
                df[node_i.name] = node_i.instantiate_values()
                print(node_i.name, "independet", list(df.columns))
            print("----"*10+"\n")
        self.df = df
        # return self.df


    def save_to_disc(self, file_name_with_path, excluded_cols=[], shorten_num_cols=True):
        if not os.path.exists(file_name_with_path):
            directory = os.path.dirname(file_name_with_path)
            pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
        if shorten_num_cols:
            self.df[self.num_cols] = self.df[self.num_cols].round(3)
        if excluded_cols:
            self.df.drop(columns=excluded_cols).to_csv(file_name_with_path, index=False)
        else:
            self.df.to_csv(file_name_with_path, index=False)
        print('--> Generated data is saved to ', file_name_with_path, '\n')


if __name__ == '__main__':
    # initialize nodes
    total_n = 100
    node_g = CategoricalNode("G", {"M": 0.5, "F": 0.5}, sample_n=total_n)

    node_u = ParetoNode("U", sample_n=total_n, shape=2.0, scale=1.0) # an unobserved pareto node
    node_x = ParetoNode("X")
    node_y = ParetoNode("Y")

    edge_g_x = CtoN("G", "X", {"M": ["Pareto", 3.0, 1.0], "F": ["Pareto", 1.0, 1.0]})
    edge_g_y = CtoN("G", "Y", {"M": ["Pareto", 3.0, 1.0], "F": ["Pareto", 2.0, 1.0]})

    edge_u_x = NtoNLinear("U", "X")
    edge_x_y = NtoNLinear("X", "Y")

    # define DAG
    nodes = [node_g, node_u, node_x, node_y]
    edge_relations = {"X": ([edge_g_x, edge_u_x], [0.5, 0.5]),
                      "Y": ([edge_g_y, edge_x_y], [0.4, 0.6])}

    mirror = Mirror(seed=0)
    mirror.generate_csv(nodes, edge_relations)
    mirror.save_to_disc("../out/synthetic_data/test/R_pareto.csv")