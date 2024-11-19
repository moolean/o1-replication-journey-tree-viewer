import pickle
from pathlib import Path
import textwrap

import streamlit as st
from streamlit_flow import streamlit_flow
from streamlit_flow.elements import StreamlitFlowEdge, StreamlitFlowNode
from streamlit_flow.layouts import TreeLayout
from streamlit_flow.state import StreamlitFlowState

from o1_replication_journey.step import STEPPER_MESSAGE, ReasoningStep, Root, Step


@st.cache_resource
def load_reasoning_tree(file_path):
    """Load a pickle file containing a reasoning tree."""
    with open(file_path, "rb") as f:
        return pickle.load(f)


def get_node_id(step: Step):
    return str(id(step))


def has_accepted_leafs(step: Step):
    if step.child_steps:
        if hasattr(step, "aborted"):
            if step.aborted:
                return False
        return any(has_accepted_leafs(child) for child in step.child_steps)
    else:
        if hasattr(step, "aborted"):
            return not step.aborted
        else:
            return step.is_terminal
        
        
def get_min_max_scores(tree: Step):
    if tree.child_steps:
        child_min_max_scores = list(zip(*[get_min_max_scores(child) for child in tree.child_steps]))
        child_min_score = min(child_min_max_scores[0])
        child_max_score = max(child_min_max_scores[1])
        return min(tree.score, child_min_score), max(tree.score, child_max_score)
    else:
        return tree.score, tree.score


def create_flow_elements(step: Step, min_score: float, max_score: float):
    """Convert a step tree into flow elements (nodes and edges)."""
    nodes = []
    edges = []
    ids_to_step = {}

    # Create unique ID for this node based on the step's hash
    node_id = get_node_id(step)
    ids_to_step[node_id] = step
    # Create node data
    if isinstance(step, Root):
        node = StreamlitFlowNode(
            id=node_id,
            pos=(0, 0),
            data={
                "content": "🌳 Root<br>\n"
                + textwrap.shorten(step.base_conversation.messages[0].content, 100)
            },
            node_type="default",
            style={
                "background": "#f5f5f5",
                "border": "3px solid #ddd",
                "borderRadius": "5px",
                "padding": "10px",
                "width": 200,
            },
        )
    else:
        if step.score == 0.0:
            score_color = "#00FF00"
        else:
            if max_score == min_score:
                weight = 0.5
            else:
                weight = (step.score - min_score) / (max_score - min_score)
            score_color = (
                "#"
                + hex(int(40+200 * (1 - weight)))[2:].zfill(2)
                + hex(int(40+200 * weight))[2:].zfill(2)
                + hex(40)[2:].zfill(2)
            )
        if step.improved_step is not None:
            improved_color = "#888888"
        else:
            improved_color = "#EEEEEE"
        node = StreamlitFlowNode(
            id=node_id,
            pos=(0, 0),
            data={
                "content": f"📝 Step (Score: {step.score:.2f})<br>\n"
                + textwrap.shorten(step.step_message.content, 100)
            },
            node_type="default",
            style={
                "background": score_color,
                "border": f"3px solid {improved_color}",
                "borderRadius": "5px",
                "padding": "10px",
                "width": 200,
            },
        )

    nodes.append(node)
    
    # Process child nodes
    for child in step.child_steps:
        # Create edge from parent to child
        child_nodes, child_edges, child_ids_to_step = create_flow_elements(
            child, min_score, max_score
        )
        child_id = child_nodes[0].id  # Get ID of first child node

        if has_accepted_leafs(child):
            edge_color = "#33FF33"
            stroke_width = "4px"
            z_index=10
        else:
            edge_color = "#555555"
            stroke_width = "2px"
            z_index=0
        edge = StreamlitFlowEdge(
            id=f"e{node_id}-{child_id}",
            source=node_id,
            target=child_id,
            edge_type="smoothstep",
            style={"stroke": edge_color, "stroke-width": stroke_width},
            z_index=z_index,
            animated=False,
        )

        edges.append(edge)

        # Add child elements
        nodes.extend(child_nodes)
        edges.extend(child_edges)
        ids_to_step.update(child_ids_to_step)

    # # Add edges for improved steps
    # if isinstance(step, ReasoningStep) and step.improved_step:
    #     improved_step_id = get_node_id(step.improved_step)
    #     print(f"Adding edge from {node_id} to {improved_step_id}")
    #     edge = StreamlitFlowEdge(
    #         id=f"e{node_id}-{improved_step_id}",
    #         source=node_id,
    #         source_position="right",
    #         target=improved_step_id,
    #         target_position="left",
    #         edge_type="smoothstep",
    #         animated=True,
    #         style={"stroke": "#888888"},
    #     )
    #     edges.append(edge)
    return nodes, edges, ids_to_step


def main():
    st.set_page_config(layout="wide")
    st.title("Reasoning Tree Flow Visualizer")

    # Get list of YAML files
    reasoning_tree_dir = Path("aime/")
    reasoning_tree_files = list(reasoning_tree_dir.glob("**/*.pkl"))

    # File selector
    selected_file = st.selectbox(
        "Select a reasoning tree file",
        reasoning_tree_files,
        format_func=lambda x: x.name,
    )

    if selected_file:
        # Load tree
        tree = load_reasoning_tree(selected_file)

        min_score, max_score = get_min_max_scores(tree)
        nodes, edges, ids_to_step = create_flow_elements(tree, min_score, max_score)

        # Create state
        state = StreamlitFlowState(nodes=nodes, edges=edges)

        collapsed_details = st.session_state.get("collapsed_details", True)
        if collapsed_details:
            column_spec = [0.8, 0.2]
        else:
            column_spec = [0.5, 0.5]
        graph_col, details_col = st.columns(column_spec)

        with graph_col:
            # Display flow diagram
            new_state = streamlit_flow(
                # NOTE: streamlit-flow bug that does not update with a new state otherwise.
                key="reasoning_tree:" + str(selected_file),
                state=state,
                height=512 + 128,
                fit_view=True,
                show_controls=True,
                pan_on_drag=True,
                allow_zoom=True,
                get_node_on_click=True,
                layout=TreeLayout(direction="down", node_node_spacing=100),
            )

        with details_col:
            st.toggle("Collapse Details", key="collapsed_details", value=collapsed_details)
            with st.container(border=True, height=512 + 128 - 40):
                # Display detailed information when a node is clicked
                if new_state.selected_id:
                    nodes = [n for n in nodes if n.id == new_state.selected_id]
                    if nodes:
                        node = nodes[0]
                        step = ids_to_step[node.id]
                        st.markdown("### Details")
                        if isinstance(step, Root):
                            st.markdown("#### Base Conversation")
                            for msg in step.base_conversation.messages:
                                st.markdown(f"- {msg.role}: {msg.content}")
                        else:
                            if step.aborted:
                                st.markdown("#### Unexplored Step")
                            else:
                                st.markdown("#### Explored Step")
                            st.markdown(step.step_message.content)
                            if isinstance(step, ReasoningStep):
                                st.markdown(f"#### Score: {step.score}")
                                st.markdown("#### Verification")
                                st.markdown(step.verification_message.content)
                                st.markdown("#### Score Reasoning")
                                st.markdown(step.score_message.content)
                                if step.terminal_message_check is not None:
                                    st.markdown("#### Terminal Check")
                                    st.markdown(step.terminal_message_check.content)
                                
                    else:
                        st.markdown("#### Node Not Found")


if __name__ == "__main__":
    main()
