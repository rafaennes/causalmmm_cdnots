"""
Benchmark CausalMMM vs Google Meridian - FIXED VERSION
======================================================

Comparison following PyMC-Labs methodology.
"""

import numpy as np
import pandas as pd
import time
import os
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

# CausalMMM
from causalmmm import CausalMMM, CausalMMMConfig
from causalmmm.preprocessing import PanelDataLoader, GroupStandardizer, temporal_train_test_split
from causalmmm.metrics.evaluation import evaluate_forecast

# Meridian (if available)
try:
    import meridian
    MERIDIAN_AVAILABLE = True
except ImportError:
    MERIDIAN_AVAILABLE = False
    print("Warning: Google Meridian not installed. Skipping Meridian benchmark.")


class BenchmarkSuite:
    """Comprehensive benchmark suite for MMM models."""
    
    def __init__(self, output_dir: str = './benchmark_results'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.results = {}
    
    def generate_synthetic_data(
        self,
        n_entities: int = 20,
        n_timesteps: int = 104,
        n_channels: int = 5,
        seed: int = 42
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """Generate synthetic MMM data with known causal structure."""
        rng = np.random.default_rng(seed)
        
        # True causal structure
        d = n_channels
        true_graph = np.zeros((d + 1, d + 1))
        true_graph[0, 2] = 1  # TV -> Digital
        true_graph[0, 5] = 1  # TV -> Sales
        true_graph[1, 5] = 1  # Radio -> Sales
        true_graph[2, 5] = 1  # Digital -> Sales
        true_graph[3, 5] = 1  # Social -> Sales
        
        rows = []
        channel_names = ['TV', 'Radio', 'Digital', 'Social', 'Search']
        
        for entity_id in range(n_entities):
            baseline_sales = 100 + rng.normal(0, 20)
            adstock = np.zeros(n_channels)
            sales = baseline_sales
            
            for t in range(n_timesteps):
                week = t % 52
                is_holiday = int(week >= 48)
                seasonal_factor = 1.0 + 0.2 * np.sin(2 * np.pi * week / 52)
                if is_holiday:
                    seasonal_factor *= 1.3
                
                base_budget = 50 + 10 * rng.normal(0, 1)
                spend = rng.lognormal(0, 0.5, size=n_channels)
                spend = spend / spend.sum() * base_budget
                
                # Causal effects
                digital_boost = 0.3 * spend[0]
                spend[2] += digital_boost
                
                saturated_spend = spend / (1 + 0.05 * spend)
                adstock = 0.6 * adstock + saturated_spend
                
                tv_effect = 0.4 * adstock[0]
                radio_effect = 0.3 * adstock[1]
                digital_effect = 0.35 * adstock[2]
                social_effect = 0.25 * adstock[3]
                search_effect = 0.05 * adstock[4]
                
                marketing_effect = (tv_effect + radio_effect + digital_effect + 
                                   social_effect + search_effect)
                
                sales = (baseline_sales * seasonal_factor +
                        marketing_effect +
                        0.3 * sales +
                        rng.normal(0, 5))
                sales = max(sales, 10)
                
                row = {
                    'entity': f'region_{entity_id}',
                    'time': t,
                    'sales': sales,
                    'is_holiday': is_holiday,
                    'week': week
                }
                for j, name in enumerate(channel_names):
                    row[name] = spend[j]
                
                rows.append(row)
        
        df = pd.DataFrame(rows)
        return df, true_graph
    
    def run_causalmmm(
        self,
        train_data: Tuple,
        test_data: Tuple,
        config: CausalMMMConfig,
        epochs: int = 50
    ) -> Dict:
        """Run CausalMMM and collect metrics."""
        print("\n" + "="*60)
        print("Running CausalMMM")
        print("="*60)
        
        X_train, y_train, time_train, context_train = train_data
        X_test, y_test, time_test, context_test = test_data
        
        # Train
        model = CausalMMM(config, context_dim=context_train.shape[1] if context_train is not None else 0)
        
        start_time = time.time()
        model.fit(X_train, y_train, context_train, time_train, epochs=epochs, verbose=1)
        train_time = time.time() - start_time
        
        # Predict
        start_time = time.time()
        y_pred = model.predict(X_test, context_test, time_test)
        inference_time = time.time() - start_time
        
        # Get causal graph
        graph = model.get_causal_graph(X_train, y_train, context_train, time_train, threshold=0.3)
        
        # Compute metrics
        y_true = y_test[:, :, 0]
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
        smape = np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)) * 100
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        mae = np.mean(np.abs(y_true - y_pred))
        
        return {
            'model': model,
            'predictions': y_pred,
            'graph': graph,
            'metrics': {
                'mape': mape,
                'smape': smape,
                'rmse': rmse,
                'mae': mae
            },
            'timing': {
                'train_time': train_time,
                'inference_time': inference_time
            }
        }
    
    def compare_causal_graphs(self, learned_graph: np.ndarray, true_graph: np.ndarray) -> Dict:
        """Compare learned graph with ground truth."""
        n = true_graph.shape[0]
        mask = ~np.eye(n, dtype=bool)
        
        true_flat = true_graph[mask]
        learned_flat = learned_graph[mask]
        
        tp = np.sum((true_flat == 1) & (learned_flat == 1))
        fp = np.sum((true_flat == 0) & (learned_flat == 1))
        fn = np.sum((true_flat == 1) & (learned_flat == 0))
        tn = np.sum((true_flat == 0) & (learned_flat == 0))
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        shd = np.sum(true_flat != learned_flat)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'shd': shd,
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn),
            'tn': int(tn)
        }
    
    def visualize_results(self, results: Dict):
        """Create comprehensive visualization of results."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Forecast accuracy
        ax = axes[0, 0]
        if 'CausalMMM' in results and 'metrics' in results['CausalMMM']:
            metrics = results['CausalMMM']['metrics']
            metric_names = list(metrics.keys())
            values = list(metrics.values())
            ax.bar(metric_names, values, color='steelblue', alpha=0.7)
            ax.set_ylabel('Error')
            ax.set_title('CausalMMM Forecast Accuracy')
            ax.grid(True, alpha=0.3)
        
        # 2. Timing
        ax = axes[0, 1]
        if 'CausalMMM' in results and 'timing' in results['CausalMMM']:
            timing = results['CausalMMM']['timing']
            ax.barh(['Train', 'Inference'], 
                   [timing['train_time'], timing['inference_time']],
                   color=['blue', 'green'], alpha=0.7)
            ax.set_xlabel('Time (seconds)')
            ax.set_title('Computational Efficiency')
            ax.grid(True, alpha=0.3)
        
        # 3. Graph quality
        ax = axes[0, 2]
        if 'CausalMMM' in results and 'graph_metrics' in results['CausalMMM']:
            graph_metrics = results['CausalMMM']['graph_metrics']
            metrics_names = ['Precision', 'Recall', 'F1']
            values = [graph_metrics['precision'], graph_metrics['recall'], graph_metrics['f1']]
            ax.bar(metrics_names, values, color='green', alpha=0.7)
            ax.set_ylim([0, 1])
            ax.set_title('Causal Structure Quality')
            ax.set_ylabel('Score')
            ax.grid(True, alpha=0.3)
        
        # 4. Predictions
        ax = axes[1, 0]
        if 'CausalMMM' in results and 'predictions' in results['CausalMMM']:
            y_pred = results['CausalMMM']['predictions']
            ax.plot(y_pred[0], label='Predicted', alpha=0.7)
            ax.set_xlabel('Time')
            ax.set_ylabel('Sales')
            ax.set_title('CausalMMM: Predictions (Entity 0)')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 5. Learned graph
        ax = axes[1, 1]
        if 'CausalMMM' in results and 'graph' in results['CausalMMM']:
            graph = results['CausalMMM']['graph']
            im = ax.imshow(graph, cmap='Blues', aspect='auto')
            ax.set_title('Learned Causal Graph')
            ax.set_xlabel('To Variable')
            ax.set_ylabel('From Variable')
            plt.colorbar(im, ax=ax)
        
        # 6. True graph
        ax = axes[1, 2]
        if 'true_graph' in results:
            true_graph = results['true_graph']
            im = ax.imshow(true_graph, cmap='Greens', aspect='auto')
            ax.set_title('True Causal Graph')
            ax.set_xlabel('To Variable')
            ax.set_ylabel('From Variable')
            plt.colorbar(im, ax=ax)
        
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'benchmark_comparison.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nVisualization saved to {save_path}")
        plt.show()
    
    def run_full_benchmark(self, save_results: bool = True):
        """Run complete benchmark suite."""
        print("="*80)
        print("CAUSALMMM BENCHMARK")
        print("="*80)
        
        # 1. Generate data
        print("\n1. Generating synthetic data...")
        df, true_graph = self.generate_synthetic_data(
            n_entities=20,
            n_timesteps=104,
            n_channels=5,
            seed=42
        )
        print(f"   Generated {len(df)} observations")
        
        # 2. Load and preprocess
        print("\n2. Loading and preprocessing...")
        loader = PanelDataLoader()
        X, y, context, entities, time_idx = loader.load_from_dataframe(
            df,
            entity_col='entity',
            time_col='time',
            channel_cols=['TV', 'Radio', 'Digital', 'Social', 'Search'],
            target_col='sales',
            context_cols=['is_holiday']
        )
        
        # Split
        train_data, val_data, test_data = temporal_train_test_split(
            X, y, time_idx, context,
            train_frac=0.7,
            val_frac=0.15
        )
        
        # Normalize
        scaler = GroupStandardizer()
        X_train, y_train, time_train, context_train = train_data
        scaler.fit(X_train, y_train, entities)
        X_train_scaled, y_train_scaled = scaler.transform(X_train, y_train, entities)
        
        X_test, y_test, time_test, context_test = test_data
        X_test_scaled, y_test_scaled = scaler.transform(X_test, y_test, entities)
        
        train_data_scaled = (X_train_scaled, y_train_scaled, time_train, context_train)
        test_data_scaled = (X_test_scaled, y_test_scaled, time_test, context_test)
        
        # 3. Run CausalMMM
        config = CausalMMMConfig(
            n_channels=5,
            lambda_kl=1.0,
            lambda_dag=1.0,
            temperature=1.0,
            temperature_min=0.5
        )
        
        causalmmm_results = self.run_causalmmm(
            train_data_scaled,
            test_data_scaled,
            config,
            epochs=30
        )
        
        # 4. Compare graphs
        if 'graph' in causalmmm_results:
            graph_metrics = self.compare_causal_graphs(
                causalmmm_results['graph'],
                true_graph
            )
            causalmmm_results['graph_metrics'] = graph_metrics
            
            print("\n" + "="*60)
            print("Causal Graph Quality")
            print("="*60)
            print(f"Precision: {graph_metrics['precision']:.3f}")
            print(f"Recall: {graph_metrics['recall']:.3f}")
            print(f"F1 Score: {graph_metrics['f1']:.3f}")
            print(f"SHD: {graph_metrics['shd']}")
        
        # 5. Compile results
        results = {
            'CausalMMM': causalmmm_results,
            'true_graph': true_graph,
            'data_info': {
                'n_entities': len(entities),
                'n_timesteps': X.shape[1],
                'n_channels': X.shape[2]
            }
        }
        
        # 6. Visualize
        print("\n" + "="*60)
        print("Generating visualizations...")
        print("="*60)
        self.visualize_results(results)
        
        # 7. Save results - LINHA 451 CORRIGIDA
        if save_results:
            import json
            results_path = os.path.join(self.output_dir, 'benchmark_results.json')
            with open(results_path, 'w') as f:
                results_json = {
                    'CausalMMM': {
                        'metrics': causalmmm_results['metrics'],
                        'timing': causalmmm_results['timing'],
                        'graph_metrics': causalmmm_results.get('graph_metrics', {})
                    },
                    'data_info': results['data_info']
                }
                json.dump(results_json, f, indent=2)
            print(f"\nResults saved to {results_path}")  # ✅ CORRIGIDO!
        
        # 8. Summary
        print("\n" + "="*80)
        print("BENCHMARK SUMMARY")
        print("="*80)
        print("\nCausalMMM Performance:")
        print(f"  MAPE: {causalmmm_results['metrics']['mape']:.2f}%")
        print(f"  SMAPE: {causalmmm_results['metrics']['smape']:.2f}%")
        print(f"  RMSE: {causalmmm_results['metrics']['rmse']:.2f}")
        print(f"  Training Time: {causalmmm_results['timing']['train_time']:.2f}s")
        
        if 'graph_metrics' in causalmmm_results:
            print(f"\nCausal Discovery:")
            print(f"  Precision: {causalmmm_results['graph_metrics']['precision']:.3f}")
            print(f"  Recall: {causalmmm_results['graph_metrics']['recall']:.3f}")
            print(f"  F1 Score: {causalmmm_results['graph_metrics']['f1']:.3f}")
        
        return results


if __name__ == "__main__":
    # Run benchmark
    benchmark = BenchmarkSuite(output_dir='./benchmark_results')
    results = benchmark.run_full_benchmark(save_results=True)
    
    print("\n✅ Benchmark completed successfully!")