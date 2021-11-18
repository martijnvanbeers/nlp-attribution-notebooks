import colorsys
import networkx as nx
import numpy as np

def get_adjmat(mat, input_tokens):
    n_layers, length, _ = mat.shape
    adj_mat = np.zeros(((n_layers+1)*length, (n_layers+1)*length))
    labels_to_index = {
            f"{k}_{input_tokens[k]}": k
                for k in range(length)
        }

    for i in np.arange(n_layers) + 1:
        labels_to_index.update({
                f"L{i}_{k}": (i * length) + k
                    for k in range(length)
            })

        adj_mat[i*length:(i+1)*length,(i-1)*length:i*length] = mat[i-1]

    return adj_mat, labels_to_index

def draw_attention_graph(adjmat, labels_to_index, n_layers, length, draw_edge_labels=False, limits=None):
    # apparently if you give a complex type to a scalar input value,
    # the numpy array constructor uses it for all the subtypes
    A = np.array(adjmat, dtype=[('weight', 'f4'), ('capacity', 'f4')])
    G=nx.from_numpy_matrix(A, create_using=nx.DiGraph())

    pos = {}
    label_pos = {}
    for i in np.arange(n_layers+1):
        for k_f in np.arange(length):
            pos[      i*length+k_f] = ((i+0.4)*2, length - k_f)
            label_pos[i*length+k_f] = (i*2      , length - k_f)

    index_to_labels = {
            value: key.split("_")[-1]
                for key, value in labels_to_index.items()
                    if value < length
        }

    nx.draw_networkx_nodes(G,pos,node_color='green', node_size=50)
    nx.draw_networkx_labels(G,pos=label_pos, labels=index_to_labels, font_size=18)

    edges, edge_widths, edge_colors = zip(*[
            (
                (node1, node2),
                attr['weight'] + 0.5,
                colorsys.hsv_to_rgb(
                    # hue; pick something blue
                    0.62,
                    # saturation, based on weight, but
                    # remapped to not go to far into grey
                    (attr['weight'] + 0.5) / 1.5,
                    # value, also based on weight and remapped,
                    # but also reversed. higher weight means
                    # lower value (=darker)
                    (-attr['weight'] + 4) / 4
                ),
            )
                for node1, node2, attr in G.edges(data=True)
                    if limits is None or (
                        isinstance(limits, tuple) and
                        len(limits) == 2 and
                        attr['weight'] >= limits[0] and
                        attr['weight'] <= limits[1]
                    )
        ])
    nx.draw_networkx_edges(
            G,
            pos,
            edgelist=edges,
            width=edge_widths,
            arrowsize=5, # smaller than default arrows
            edge_color=edge_colors
        )
    if draw_edge_labels:
        edge_labels = {
                (node1, node2): str(round(attr['capacity'],3))
                    for node1, node2, attr in G.edges(data=True)
                        if limits is None or (
                            isinstance(limits, tuple) and
                            len(limits) == 2 and
                            attr['weight'] >= limits[0] and
                            attr['weight'] <= limits[1]
                        )
            }
        nx.draw_networkx_edge_labels(G,pos,edge_labels, label_pos=0.2)

    return G

def compute_flows(G, labels_to_index, input_nodes, length):
    number_of_nodes = len(labels_to_index)
    flow_values=np.zeros((number_of_nodes,number_of_nodes))
    for key in labels_to_index:
        if key not in input_nodes:
            current_layer = int(labels_to_index[key] / length)
            pre_layer = current_layer - 1
            u = labels_to_index[key]
            for inp_node_key in input_nodes:
                v = labels_to_index[inp_node_key]
                flow_value = nx.maximum_flow_value(G,u,v, flow_func=nx.algorithms.flow.edmonds_karp)
                flow_values[u][pre_layer*length+v ] = flow_value
            flow_values[u] /= flow_values[u].sum()

    return flow_values

def compute_node_flow(G, labels_to_index, input_nodes, output_nodes,length):
    number_of_nodes = len(labels_to_index)
    flow_values=np.zeros((number_of_nodes,number_of_nodes))
    for key in output_nodes:
        if key not in input_nodes:
            current_layer = int(labels_to_index[key] / length)
            pre_layer = current_layer - 1
            u = labels_to_index[key]
            for inp_node_key in input_nodes:
                v = labels_to_index[inp_node_key]
                flow_value = nx.maximum_flow_value(G,u,v, flow_func=nx.algorithms.flow.edmonds_karp)
                flow_values[u][pre_layer*length+v ] = flow_value
            flow_values[u] /= flow_values[u].sum()

    return flow_values

def compute_joint_attention(att_mat, add_residual=True):
    if add_residual:
        residual_att = np.eye(att_mat.shape[-1])[None,...]
        aug_att_mat = att_mat + residual_att
        aug_att_mat = aug_att_mat / aug_att_mat.sum(axis=-1)[...,None]
    else:
       aug_att_mat =  att_mat

    joint_attentions = np.zeros(aug_att_mat.shape)

    layers = joint_attentions.shape[0]
    joint_attentions[0] = aug_att_mat[0]
    for i in np.arange(1,layers):
        # matmul does dot product on the last two axes,
        joint_attentions[i] = np.matmul(aug_att_mat[i], joint_attentions[i-1])

    return joint_attentions
