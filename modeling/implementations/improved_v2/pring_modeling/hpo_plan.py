from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Iterable

import pandas as pd


def _parse_values(text: str, cast=str):
    return [cast(x) for x in str(text).replace(',', ' ').split() if str(x).strip()]


def run(args: argparse.Namespace) -> dict:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = _parse_values(args.seeds, int)
    rows = []
    if args.stage in {'stage3_rgcn', 'all'}:
        for seed, hd, nl, drop, lr, neigh, bs, bpr in itertools.product(
            seeds,
            _parse_values(args.hidden_dims, int),
            _parse_values(args.num_layers, int),
            _parse_values(args.dropouts, float),
            _parse_values(args.learning_rates, float),
            _parse_values(args.neighbor_sets, str),
            _parse_values(args.batch_sizes, int),
            _parse_values(args.bpr_weights, float),
        ):
            name = f"rgcn_s{seed}_h{hd}_l{nl}_d{drop}_lr{lr}_n{neigh.replace(',', '-')}_b{bs}_bpr{bpr}".replace('.', 'p')
            cmd = (
                f"MODEL_IMPL=improved_v2 RUN_STAGE1=false RUN_STAGE2=false RUN_STAGE3_RGCN=true RUN_STAGE3_HGT=false RUN_ENSEMBLE=false RUN_COMPARE=true "
                f"MODEL_RGCN_HIDDEN_DIM={hd} MODEL_RGCN_NUM_LAYERS={nl} MODEL_RGCN_BATCH_SIZE={bs} MODEL_RGCN_NUM_NEIGHBORS='{neigh}' "
                f"MODEL_DROPOUT={drop} MODEL_LR={lr} MODEL_BPR_WEIGHT={bpr} MODEL_SEED={seed} "
                f"MODEL_OUTPUT_DIR='{args.project_dir}/models_hpo_improved_v2/{name}' MODEL_REPORT_DIR='{args.project_dir}/reports/hpo_improved_v2/{name}' "
                f"sbatch modeling/scripts/run_all_models_compare_hpc.sh"
            )
            rows.append({'stage': 'stage3_rgcn', 'name': name, 'seed': seed, 'hidden_dim': hd, 'num_layers': nl, 'dropout': drop, 'lr': lr, 'neighbors': neigh, 'batch_size': bs, 'bpr_weight': bpr, 'command': cmd})
    if args.stage in {'stage3_hgt', 'all'}:
        for seed, hd, nl, heads, drop, lr, neigh, bs, bpr in itertools.product(
            seeds,
            _parse_values(args.hgt_hidden_dims, int),
            _parse_values(args.num_layers, int),
            _parse_values(args.hgt_heads, int),
            _parse_values(args.dropouts, float),
            _parse_values(args.learning_rates, float),
            _parse_values(args.hgt_neighbor_sets, str),
            _parse_values(args.hgt_batch_sizes, int),
            _parse_values(args.bpr_weights, float),
        ):
            name = f"hgt_s{seed}_h{hd}_l{nl}_heads{heads}_d{drop}_lr{lr}_n{neigh.replace(',', '-')}_b{bs}_bpr{bpr}".replace('.', 'p')
            cmd = (
                f"MODEL_IMPL=improved_v2 RUN_STAGE1=false RUN_STAGE2=false RUN_STAGE3_RGCN=false RUN_STAGE3_HGT=true RUN_ENSEMBLE=false RUN_COMPARE=true "
                f"MODEL_HGT_HIDDEN_DIM={hd} MODEL_HGT_NUM_LAYERS={nl} MODEL_HGT_HEADS={heads} MODEL_HGT_BATCH_SIZE={bs} MODEL_HGT_NUM_NEIGHBORS='{neigh}' "
                f"MODEL_DROPOUT={drop} MODEL_LR={lr} MODEL_BPR_WEIGHT={bpr} MODEL_SEED={seed} "
                f"MODEL_OUTPUT_DIR='{args.project_dir}/models_hpo_improved_v2/{name}' MODEL_REPORT_DIR='{args.project_dir}/reports/hpo_improved_v2/{name}' "
                f"sbatch modeling/scripts/run_all_models_compare_hpc.sh"
            )
            rows.append({'stage': 'stage3_hgt', 'name': name, 'seed': seed, 'hidden_dim': hd, 'num_layers': nl, 'heads': heads, 'dropout': drop, 'lr': lr, 'neighbors': neigh, 'batch_size': bs, 'bpr_weight': bpr, 'command': cmd})
    if args.stage in {'stage2', 'all'}:
        for seed, model, dim, neg, repeat in itertools.product(seeds, _parse_values(args.kge_models), _parse_values(args.kge_dims, int), _parse_values(args.kge_negatives, int), _parse_values(args.kge_target_repeats, int)):
            name = f"kge_{model}_s{seed}_d{dim}_neg{neg}_rep{repeat}"
            cmd = (
                f"MODEL_IMPL=improved_v2 RUN_STAGE1=false RUN_STAGE2=true RUN_STAGE3_RGCN=false RUN_STAGE3_HGT=false RUN_ENSEMBLE=false RUN_COMPARE=true "
                f"MODEL_STAGE2_MODELS='{model}' MODEL_KGE_DIM={dim} MODEL_STAGE2_NEGATIVES_PER_POSITIVE={neg} MODEL_STAGE2_TARGET_TRAIN_REPEAT={repeat} MODEL_SEED={seed} "
                f"MODEL_OUTPUT_DIR='{args.project_dir}/models_hpo_improved_v2/{name}' MODEL_REPORT_DIR='{args.project_dir}/reports/hpo_improved_v2/{name}' "
                f"sbatch modeling/scripts/run_all_models_compare_hpc.sh"
            )
            rows.append({'stage': 'stage2', 'name': name, 'seed': seed, 'model': model, 'dim': dim, 'negatives': neg, 'target_repeat': repeat, 'command': cmd})
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / 'hpo_plan.csv', index=False)
    with (out_dir / 'submit_hpo.sh').open('w', encoding='utf-8') as f:
        f.write('#!/usr/bin/env bash\nset -euo pipefail\ncd "{}"\n'.format(args.project_dir))
        for cmd in df['command'].head(args.max_jobs if args.max_jobs > 0 else len(df)):
            f.write(cmd + '\n')
    return {'status': 'created', 'rows': int(len(df)), 'hpo_plan_csv': str(out_dir / 'hpo_plan.csv'), 'submit_script': str(out_dir / 'submit_hpo.sh')}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Create an HPO/seed sweep plan for finalized PRING improved_v2 models.')
    p.add_argument('--output-dir', required=True)
    p.add_argument('--project-dir', default='/home/asmaaali/KG-ADMET-Predictor')
    p.add_argument('--stage', choices=['stage2', 'stage3_rgcn', 'stage3_hgt', 'all'], default='all')
    p.add_argument('--seeds', default='1 2 3 4 5')
    p.add_argument('--hidden-dims', default='64 128')
    p.add_argument('--hgt-hidden-dims', default='64 128')
    p.add_argument('--num-layers', default='2 3')
    p.add_argument('--hgt-heads', default='2 4')
    p.add_argument('--dropouts', default='0.1 0.2 0.3')
    p.add_argument('--learning-rates', default='0.0003 0.001')
    p.add_argument('--neighbor-sets', default='15,10 20,10')
    p.add_argument('--hgt-neighbor-sets', default='10,5 15,10')
    p.add_argument('--batch-sizes', default='64 128')
    p.add_argument('--hgt-batch-sizes', default='32 64')
    p.add_argument('--bpr-weights', default='0.1 0.3 0.5')
    p.add_argument('--kge-models', default='complex')
    p.add_argument('--kge-dims', default='64 128')
    p.add_argument('--kge-negatives', default='2 5 10')
    p.add_argument('--kge-target-repeats', default='10 20 30')
    p.add_argument('--max-jobs', type=int, default=0)
    return p


def main(argv: Iterable[str] | None = None) -> None:
    print(json.dumps(run(build_parser().parse_args(argv)), indent=2))


if __name__ == '__main__':
    main()
