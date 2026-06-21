"""Integrated PRING CYP450 modeling workflows.

Modules:
- stage1_tabular: memory-safe Stage 1 baseline from PRING pair features.
- stage1_mlp_gds: optional PyTorch MLP over Neo4j GDS embeddings.
- stage2_kge: DistMult/ComplEx/RotatE KG embedding baselines.
- stage3_rgcn and stage3_hgt: heterogeneous GNN models.
- stage4_explain: evidence-path explanations for predictions.
- run_all: Docker-friendly orchestrator.
"""

__version__ = "0.2.0"
