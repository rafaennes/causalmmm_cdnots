"""
INTEGRATION GUIDE: Adding CD-NOTS arms to run_benchmark.py
===========================================================

This file shows the exact code snippets to add to run_benchmark.py
to enable the CD-NOTS benchmark arms (3 and 4).

STEP 1: Add --cdnots flag to argparse (in parse_args function)
STEP 2: Add CD-NOTS imports 
STEP 3: Add discovery + fitting phase in run_benchmark_for_dataset
STEP 4: Add evaluation for CD-NOTS models

Dependencies to install:
    pip install causal-learn scikit-learn
"""

# ============================================================
# STEP 1: Add to parse_args() after the existing arguments
# ============================================================
ARGPARSE_ADDITION = '''
    parser.add_argument(
        "--cdnots",
        action="store_true",
        help="Run CD-NOTS causal discovery and fit graph-informed models"
    )
    parser.set_defaults(cdnots=False)
'''

# ============================================================
# STEP 2: Add imports at top of run_benchmark.py
# ============================================================
IMPORT_ADDITION = '''
from mmm_param_recovery.benchmarking import cdnots_discovery
from mmm_param_recovery.benchmarking import cdnots_fitter
from mmm_param_recovery.benchmarking import cdnots_model_builder
'''

# ============================================================
# STEP 3: Add inside run_benchmark_for_dataset(), after the
# existing "Fit PyMC models" block (still inside PHASE 1)
# ============================================================
PHASE1_ADDITION = '''
        # ============================================================
        # CD-NOTS ARMS: Discover graph + fit graph-informed models
        # ============================================================
        if getattr(args, 'cdnots', False):
            console.print()
            console.rule("[bold magenta]PHASE 1b: CD-NOTS CAUSAL DISCOVERY + FITTING[/bold magenta]")
            
            # Phase 0: Discover causal graph (runs once per dataset)
            graph = cdnots_discovery.discover_graph(
                data_df, channel_columns,
                alpha=0.05, max_lag=2, console=console
            )
            
            # Save graph for reproducibility
            import pickle
            graph_path = Path(f"data/results/{dataset_name}/cdnots_graph.pkl")
            with open(graph_path, 'wb') as f:
                pickle.dump(graph, f)
            console.print(f"  [green]✓[/green] Saved CD-NOTS graph to {graph_path}")
            
            # Arm 3: PyMC + CD-NOTS
            if "pymc" in args.libraries:
                for sampler in args.samplers:
                    if model_fitter.should_skip_sampler(sampler, dataset_name, console):
                        continue
                    
                    cdnots_key = f"cdnots_pymc_{sampler}"
                    if not storage.model_exists(dataset_name, cdnots_key) or args.force_rerun:
                        console.print(f"\\n[bold yellow]--- PyMC + CD-NOTS - {sampler} ---[/bold yellow]")
                        try:
                            pymc_cdnots, runtime, ess = cdnots_fitter.fit_pymc_with_graph(
                                data_df, channel_columns, control_columns, graph,
                                sampler, args.chains, args.draws, args.tune,
                                args.target_accept, args.seed, console
                            )
                            storage.save_pymc_model(pymc_cdnots, dataset_name, f"cdnots_{sampler}", runtime, ess)
                            del pymc_cdnots
                            gc.collect()
                        except Exception as e:
                            console.print(f"  [red]✗[/red] PyMC + CD-NOTS - {sampler} failed: {e}")
            
            # Arm 4: Meridian + CD-NOTS
            if "meridian" in args.libraries:
                if not storage.model_exists(dataset_name, "cdnots_meridian") or args.force_rerun:
                    console.print(f"\\n[bold yellow]--- Meridian + CD-NOTS ---[/bold yellow]")
                    try:
                        meridian_cdnots, runtime, ess = cdnots_fitter.fit_meridian_with_graph(
                            data_df, channel_columns, control_columns, graph,
                            args.chains, args.draws, args.tune,
                            args.target_accept, args.seed, console
                        )
                        storage.save_meridian_model(meridian_cdnots, dataset_name + "_cdnots", runtime, ess)
                        del meridian_cdnots
                        gc.collect()
                    except Exception as e:
                        console.print(f"  [red]✗[/red] Meridian + CD-NOTS failed: {e}")
'''

# ============================================================
# STEP 4: Add inside PHASE 2 (evaluation), after existing
# PyMC evaluation block
# ============================================================
PHASE2_ADDITION = '''
    # Evaluate CD-NOTS models
    if getattr(args, 'cdnots', False):
        console.print()
        console.rule("[bold magenta]EVALUATING CD-NOTS MODELS[/bold magenta]")
        
        # Evaluate PyMC + CD-NOTS
        if "pymc" in args.libraries:
            for sampler in args.samplers:
                cdnots_key = f"cdnots_{sampler}"
                if storage.model_exists(dataset_name, "pymc", cdnots_key):
                    console.print(f"\\n[bold yellow]--- Evaluating PyMC + CD-NOTS - {sampler} ---[/bold yellow]")
                    pymc_cdnots, runtime, ess = storage.load_pymc_model(dataset_name, cdnots_key)
                    model_name = f"PyMC + CD-NOTS - {sampler}"
                    results[model_name] = (pymc_cdnots, runtime, ess)
                    
                    perf_rows = evaluation.evaluate_pymc_fit(pymc_cdnots, data_df, cdnots_key)
                    for row in perf_rows:
                        row["Dataset"] = dataset_name
                    all_performance_rows.extend(perf_rows)
                    
                    channel_df, avg_metrics = evaluation.evaluate_pymc_channel_contributions(
                        pymc_cdnots, truth_df, channel_columns, cdnots_key, dataset_name
                    )
                    all_channel_contribution_rows.append(channel_df)
                    channel_contribution_averages.append({
                        "Dataset": dataset_name,
                        "Model": model_name,
                        **avg_metrics
                    })
        
        # Evaluate Meridian + CD-NOTS
        if "meridian" in args.libraries:
            cdnots_dataset = dataset_name + "_cdnots"
            if storage.model_exists(cdnots_dataset, "meridian"):
                console.print(f"\\n[bold yellow]--- Evaluating Meridian + CD-NOTS ---[/bold yellow]")
                meridian_cdnots, runtime, ess = storage.load_meridian_model(cdnots_dataset)
                model_name = "Meridian + CD-NOTS"
                results[model_name] = (meridian_cdnots, runtime, ess)
                
                perf_rows = evaluation.evaluate_meridian_fit(meridian_cdnots, data_df)
                for row in perf_rows:
                    row["Dataset"] = dataset_name
                all_performance_rows.extend(perf_rows)
                
                channel_df, avg_metrics = evaluation.evaluate_meridian_channel_contributions(
                    meridian_cdnots, truth_df, channel_columns, dataset_name
                )
                all_channel_contribution_rows.append(channel_df)
                channel_contribution_averages.append({
                    "Dataset": dataset_name,
                    "Model": model_name,
                    **avg_metrics
                })
'''


# ============================================================
# Quick test command
# ============================================================
QUICKSTART = """
# Quick test (small dataset, 1 sampler):
python run_benchmark.py --datasets small_business --samplers nutpie --chains 2 --draws 500 --tune 500 --cdnots

# Full benchmark:
python run_benchmark.py --datasets small_business medium_business --samplers nutpie --chains 4 --draws 1000 --tune 1000 --cdnots
"""

if __name__ == "__main__":
    print("=" * 60)
    print("CD-NOTS Integration Guide for run_benchmark.py")
    print("=" * 60)
    print("\nFiles to add to mmm_param_recovery/benchmarking/:")
    print("  1. cdnots_discovery.py    - Causal graph discovery")
    print("  2. cdnots_model_builder.py - Graph-informed model building")
    print("  3. cdnots_fitter.py       - Fitting with graph priors")
    print("\nModifications to run_benchmark.py:")
    print("  Step 1: Add --cdnots argparse flag")
    print("  Step 2: Add imports")
    print("  Step 3: Add PHASE 1b (discovery + fitting)")
    print("  Step 4: Add evaluation for CD-NOTS models")
    print(f"\n{QUICKSTART}")
