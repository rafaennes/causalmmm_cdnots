"""
Benchmark CausalMMM vs Google Meridian
======================================

Comparison following PyMC-Labs methodology:
https://www.pymc-labs.com/blog-posts/pymc-marketing-vs-google-meridian

Metrics:
- Forecast accuracy (MAPE, SMAPE, RMSE)
- Causal structure quality (if ground truth available)
- Attribution accuracy (channel contributions)
- Computational efficiency
- Interpretability
"""

import numpy as np
import pandas as pd
import time
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

# CausalMMM
from causalmmm import CausalMMM, CausalMMMConfig
from causalmmm.preprocessing import PanelDataLoader, GroupStandardizer, temporal_train_test_split
from causalmmm.metrics.evaluation import evaluate_forecast, compute_attribution

# Meridian (if available)
try:
    import meridian
    MERIDIAN_AVAILABLE = True
except ImportError:
    MERIDIAN_AVAILABLE = False
    print("Warning: Google Meridian not installed. Install with: pip install meridian")


class BenchmarkSuite:
    """
    Comprehensive benchmark suite for MMM models.
    """
    
    def __init__(self, output_dir: str = './benchmark_results'):
        self.output_dir = output_dir
        self.results = {}
    
    def generate_synthetic_data(
        self,
        n_entities: int = 20,
        n_timesteps: int = 104,
        n_channels: int = 5,
        seed: int = 42
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Generate synthetic MMM data with known causal structure.
        
        Returns:
            df: Panel DataFrame
            true_graph: [d+1, d+1] - True causal adjacency matrix
        """
        rng = np.random.default_rng(seed)
        
        # Define true causal structure
        # Channel 0 (TV) -> Channel 2 (Digital) -> Sales
        # Channel 1 (Radio) -> Sales
        # Channel 3 (Social) -> Sales
        # Channel 4 (Search) has weak effect
        
        d = n_channels
        true_graph = np.zeros((d + 1, d + 1))
        true_graph[0, 2] = 1  # TV -> Digital
        true_graph[0, 5] = 1  # TV -> Sales
        true_graph[1, 5] = 1  # Radio -> Sales
        true_graph[2, 5] = 1  # Digital -> Sales
        true_graph[3, 5] = 1  # Social -> Sales
        # Channel 4 (Search) has no edges
        
        rows = []
        channel_names = ['TV', 'Radio', 'Digital', 'Social', 'Search']
        
        for entity_id in range(n_entities):
            # Entity-specific baseline
            baseline_sales = 100 + rng.normal(0, 20)
            
            # State for adstock
            adstock = np.zeros(n_channels)
            sales = baseline_sales
            
            for t in range(n_timesteps):
                # Time features
                week = t % 52
                is_holiday = int(week >= 48)  # Last month
                
                # Seasonality
                seasonal_factor = 1.0 + 0.2 * np.sin(2 * np.pi * week / 52)
                if is_holiday:
                    seasonal_factor *= 1.3
                
                # Marketing spend
                base_budget = 50 + 10 * rng.normal(0, 1)
                spend = rng.lognormal(0, 0.5, size=n_channels)
                spend = spend / spend.sum() * base_budget
                
                # Apply causal structure
                # TV influences Digital
                digital_boost = 0.3 * spend[0]  # TV -> Digital
                spend[2] += digital_boost
                
                # Saturation
                saturated_spend = spend / (1 + 0.05 * spend)
                
                # Adstock (carryover)
                adstock = 0.6 * adstock + saturated_spend
                
                # Sales response (following true causal graph)
                tv_effect = 0.4 * adstock[0]
                radio_effect = 0.3 * adstock[1]
                digital_effect = 0.35 * adstock[2]
                social_effect = 0.25 * adstock[3]
                search_effect = 0.05 * adstock[4]  # Weak effect
                
                marketing_effect = (tv_effect + radio_effect + digital_effect + 
                                   social_effect + search_effect)
                
                # Sales equation
                sales = (baseline_sales * seasonal_factor +
                        marketing_effect +
                        0.3 * sales +  # Autoregressive
                        rng.normal(0, 5))  # Noise
                sales = max(sales, 10)
                
                # Record
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
    
    def run_meridian(self, train_data: Tuple, test_data: Tuple) -> Dict:
        """Run Google Meridian (if available)."""
        if not MERIDIAN_AVAILABLE:
            return {'error': 'Meridian not installed'}
        
        print("\n" + "="*60)
        print("Running Google Meridian")
        print("="*60)
        
        # TODO: Implement Meridian benchmark
        # This requires understanding Meridian's API
        
        return {'error': 'Meridian benchmark not yet implemented'}
    
    def compare_causal_graphs(self, learned_graph: np.ndarray, true_graph: np.ndarray) -> Dict:
        """
        Compare learned graph with ground truth.
        
        Metrics:
        - Precision: TP / (TP + FP)
        - Recall: TP / (TP + FN)
        - F1 Score
        - SHD (Structural Hamming Distance)
        """
        # Flatten and remove self-loops
        n = true_graph.shape[0]
        mask = ~np.eye(n, dtype=bool)
        
        true_flat = true_graph[mask]
        learned_flat = learned_graph[mask]
        
        # Binary classification metrics
        tp = np.sum((true_flat == 1) & (learned_flat == 1))
        fp = np.sum((true_flat == 0) & (learned_flat == 1))
        fn = np.sum((true_flat == 1) & (learned_flat == 0))
        tn = np.sum((true_flat == 0) & (learned_flat == 0))
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        # Structural Hamming Distance
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
        
        # 1. Forecast accuracy comparison
        ax = axes[0, 0]
        models = list(results.keys())
        metrics = ['mape', 'smape', 'rmse', 'mae']
        
        for metric in metrics:
            values = [results[m]['metrics'][metric] for m in models if 'metrics' in results[m]]
            ax.bar(np.arange(len(models)) + metrics.index(metric) * 0.2, values, 
                   width=0.2, label=metric.upper())
        
        ax.set_xticks(np.arange(len(models)) + 0.3)
        ax.set_xticklabels(models)
        ax.set_ylabel('Error')
        ax.set_title('Forecast Accuracy Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Timing comparison
        ax = axes[0, 1]
        train_times = [results[m]['timing']['train_time'] for m in models if 'timing' in results[m]]
        ax.barh(models, train_times)
        ax.set_xlabel('Training Time (seconds)')
        ax.set_title('Computational Efficiency')
        ax.grid(True, alpha=0.3)
        
        # 3. Graph structure quality (if available)
        if 'graph_metrics' in results.get('CausalMMM', {}):
            ax = axes[0, 2]
            graph_metrics = results['CausalMMM']['graph_metrics']
            metrics_names = ['Precision', 'Recall', 'F1']
            values = [graph_metrics['precision'], graph_metrics['recall'], graph_metrics['f1']]
            ax.bar(metrics_names, values)
            ax.set_ylim([0, 1])
            ax.set_title('Causal Structure Quality')
            ax.set_ylabel('Score')
            ax.grid(True, alpha=0.3)
        
        # 4. Predictions vs actual (CausalMMM)
        if 'CausalMMM' in results:
            ax = axes[1, 0]
            y_pred = results['CausalMMM']['predictions']
            # Plot first entity
            ax.plot(y_pred[0], label='Predicted', alpha=0.7)
            ax.set_xlabel('Time')
            ax.set_ylabel('Sales')
            ax.set_title('CausalMMM: Predictions (Entity 0)')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 5. Learned causal graph
        if 'graph' in results.get('CausalMMM', {}):
            ax = axes[1, 1]
            graph = results['CausalMMM']['graph']
            im = ax.imshow(graph, cmap='Blues', aspect='auto')
            ax.set_title('Learned Causal Graph')
            ax.set_xlabel('To Variable')
            ax.set_ylabel('From Variable')
            plt.colorbar(im, ax=ax)
        
        # 6. True causal graph (if available)
        if 'true_graph' in results:
            ax = axes[1, 2]
            true_graph = results['true_graph']
            im = ax.imshow(true_graph, cmap='Greens', aspect='auto')
            ax.set_title('True Causal Graph')
            ax.set_xlabel('To Variable')
            ax.set_ylabel('From Variable')
            plt.colorbar(im, ax=ax)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/benchmark_comparison.png', dpi=150, bbox_inches='tight')
        print(f"\nVisualization saved to {self.output_dir}/benchmark_comparison.png")
        plt.show()
    
    def run_full_benchmark(self, save_results: bool = True):
        """
        Run complete benchmark suite.
        """
        print("="*80)
        print("CAUSALMMM VS GOOGLE MERIDIAN BENCHMARK")
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
        print(f"   Entities: {df['entity'].nunique()}")
        print(f"   Timesteps: {df['time'].nunique()}")
        
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
        
        # 4. Run Meridian (if available)
        meridian_results = self.run_meridian(train_data, test_data)
        
        # 5. Compare graphs
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
        
        # 6. Compile results
        results = {
            'CausalMMM': causalmmm_results,
            'Meridian': meridian_results,
            'true_graph': true_graph,
            'data_info': {
                'n_entities': len(entities),
                'n_timesteps': X.shape[1],
                'n_channels': X.shape[2]
            }
        }
        
        # 7. Visualize
        print("\n" + "="*60)
        print("Generating visualizations...")
        print("="*60)
        self.visualize_results(results)
        
        # 8. Save results
        if save_results:
            import json
            with open(f'{self.output_dir}/benchmark_results.json', 'w') as f:
                # Convert numpy arrays to lists for JSON
                results_json = {
                    'CausalMMM': {
                        'metrics': causalmmm_results['metrics'],
                        'timing': causalmmm_results['timing'],
                        'graph_metrics': causalmmm_results.get('graph_metrics', {})
                    },
                    'data_info': results['data_info']
                }
                json.dump(results_json, f, indent=2)
            print(f"\nResults saved to {self.output})