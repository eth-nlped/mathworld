import graphviz
import sympy

def extract_container(c_obj):
    c_obj_dict = c_obj.get_dict()
    c_header = f"{str(c_obj_dict['id'])}: {str(c_obj_dict['label'])}"
    c_body = ""

    for k, v in c_obj.get_dict().items():
        if v == None:
            v = "-"
        if k not in ["id", "label"]:
            c_body += f"{str(k)}: {str(v)}\l"  ## left aligned

    return c_header, c_body

def add_containers(g: graphviz.Digraph, containers: dict) -> graphviz.Digraph:
    for i, (c_id, c_obj) in enumerate(containers.items()):
        c_header, c_body = extract_container(c_obj)  ## extract container content
        c_content = f'{{<{c_id}> {c_header} | {c_body}}}'  ## construct container
        g.node('struct' + str(c_id), c_content)
    return g


def extract_relation(r_obj, r_details: bool = True):
    r_obj_dict = r_obj.get_dict()
    r_src = ""
    r_tgt = ""
    r_label = ""
    if 'source' in r_obj_dict.keys():
        source_id = str(r_obj_dict['source'].id)
        r_src = f"struct{source_id}:{source_id}"
    if 'target' in r_obj_dict.keys():
        target_id = str(r_obj_dict['target'].id)
        r_tgt = f"struct{target_id}:{target_id}"
    if 'id' in r_obj_dict.keys() and 'type' in r_obj_dict.keys():
        r_label += f"{str(r_obj_dict['id'])} {str(r_obj_dict['type'])}\l"

    if r_details:
        for k, v in r_obj.get_dict().items():
            if v == None:
                v = "-"
            if k not in ["source", "target", "type", "id"]:
                r_label += f"{str(k)}: {str(v)}\l"  ## left aligned
    return r_src, r_tgt, r_label


def add_relations(g: graphviz.Digraph, relations: dict, r_details: bool = True) -> graphviz.Digraph:
    for i, (r_id, r_obj) in enumerate(relations.items()):
        r_src, r_tgt, r_label = extract_relation(r_obj, r_details)  ## extract container content
        g.edge(r_src, r_tgt, label=r_label)  ## construct container
    return g


def add_ref(g: graphviz.Digraph, ref = "") -> graphviz.Digraph:
    c_content = f'{{ref: {ref}}}' ## construct container
    g.node('struct' + str(ref), c_content)
    return g


def visualize_mwp_state(state, mwp_name: str = "mwp", file_format='png', show_plot: bool = True):
    g = graphviz.Digraph('structs', filename=str(mwp_name), node_attr={'shape': 'record'})
    g = add_containers(g, state.containers)
    g = add_relations(g, state.relations)
    g = add_ref(g, state.ref)
    g.render(f'../output_files/viz/{str(mwp_name)}.gv', format=file_format, cleanup=True,
                 view=show_plot)

def visualize_mwp_interactive(state, mwp_name: str = "mwp"):
    g = graphviz.Digraph('structs', filename=str(mwp_name), node_attr={'shape': 'record'})
    g = add_containers(g, state.containers)
    g = add_relations(g, state.relations)
    g = add_ref(g, state.ref)
    return g