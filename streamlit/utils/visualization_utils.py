import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


def visualize_graph(result):
    edges = [(record[0], record[2]) for record in result]

    G = nx.DiGraph()
    G.add_edges_from(edges)

    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_color="skyblue", node_size=2000, font_size=12, arrows=True)
    plt.title("Compound Similarity Graph")
    st.pyplot(plt)


def display_table(result):
    df = pd.DataFrame(result, columns=["Compound1", "Compound2"])
    st.dataframe(df)
