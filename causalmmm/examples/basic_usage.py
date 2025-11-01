"""
Basic Usage Example for CausalMMM
==================================

Simple end-to-end example showing how to use the library.
"""

import numpy as np
import pandas as pd
from causalmmm import CausalMMM, CausalMMMConfig
from causalmmm.preprocessing import PanelDataLoader, GroupStandardizer, temporal_train_test_split
from causalmmm.visualization import plot_causal_graph, plot_channel_contribution


def main():
    print("="*60)
    print("CausalMMM - Basic Usage Example")
    print("="*60)
    
    # 1. Generate sample data
    print("\n1. Generating sample data...")
    np.random.seed(42)
    
    entities = ['store_A', 'store_B', 'store_C', 'store_D', 'store_E']
    timesteps = 80
    channels = ['TV', 'Radio', 'Digital', 'Social']
    
    rows = []
    for entity in entities:
        baseline = 100 + np.random.normal(0, 10)
        sales = baseline
        
        for t in range(timesteps):
            # Marketing spend
            tv = np.random.lognormal(2, 0.5)
            radio = np.random.lognormal(1.5, 0.5)
            digital = np.random.lognormal(1.8, 0.5)
            social = np.random.lognormal(1.2, 0.5)
            
            # True causal structure: TV -> Digital -> Sales, Radio -> Sales
            digital = digital + 0.2 * tv  # TV influences Digital
            
            # Sales response with saturation
            tv_effect = 0.5 * tv / (1 + 0.1 * tv)
            radio_effect = 0.3 * radio / (1 + 0.1 * radio)
            digital_effect = 0.4 * digital / (1 + 0.1 * digital)
            social_effect = 0.2 * social / (1 + 0.1 * social)
            
            sales = (baseline + tv_effect + radio_effect + digital_effect + social_effect +
                    0.4 * sales + np.random.normal(0, 5))
            sales = max(sales, 10)
            
            rows.append({
                'entity': entity,
                'time': t,
                'TV': tv,
                'Radio': radio,
                'Digital': digital,
                'Social': social,
                'sales': sales,
                'is_holiday': int(t % 20 == 0)
            })
    
    df = pd.DataFrame(rows)
    print(f"   Created {len(df)} observations")
    
    # 2. Load data
    print("\n2. Loading data...")
    loader = PanelDataLoader()
    X, y, context, entities, time_idx = loader.load_from_dataframe(
        df,
        entity_col='entity',
        time_col='time',
        channel_cols=channels,
        target_col='sales',
        context_cols=['is_holiday']
    )
    print(f"   X shape: {X.shape}")
    print(f"   y shape: {y.shape}")
    
    # 3. Split data
    print("\n3. Splitting data...")
    train_data, val_data, test_data = temporal_train_test_split(
        X, y, time_idx, context,
        train_frac=0.7,
        val_frac=0.15
    )
    
    # 4. Normalize
    print("\n4. Normalizing data...")
    scaler = GroupStandardizer()
    X_train, y_train, time_train, context_train = train_data
    scaler.fit(X_train, y_train, entities)
    X_train_n, y_train_n = scaler.transform(X_train, y_train, entities)
    
    X_test, y_test, time_test, context_test = test_data
    X_test_n, y_test_n = scaler.transform(X_test, y_test, entities)
    
    # 5. Configure model
    print("\n5. Configuring model...")
    config = CausalMMMConfig(
        n_channels=len(channels),
        lambda_kl=1.0,
        lambda_dag=1.0,
        temperature=1.0,
        temperature_min=0.5,
        temperature_decay=0.995
    )
    
    # 6. Train model
    print("\n6. Training model...")
    model = CausalMMM(config, context_dim=1)
    model.fit(
        X_train_n, y_train_n, context_train, time_train,
        epochs=30,
        verbose=1
    )
    
    # 7. Predict
    print("\n7. Making predictions...")
    y_pred = model.predict(X_test_n, context_test, time_test)
    
    # Calculate metrics
    y_true = y_test_n[:, :, 0]
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    print(f"\nTest Performance:")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  RMSE: {rmse:.4f}")
    
    # 8. Extract causal graph
    print("\n8. Extracting causal graph...")
    graph = model.get_causal_graph(
        X_train_n, y_train_n, context_train, time_train,
        threshold=0.3
    )
    
    print("\nLearned Causal Graph:")
    print(graph)
    
    print("\nSignificant Edges:")
    for i, src in enumerate(channels + ['sales']):
        for j, tgt in enumerate(channels + ['sales']):
            if i != j and graph[i, j] > 0:
                print(f"  {src} → {tgt}")
    
    # 9. Visualize (if matplotlib available)
    print("\n9. Generating visualizations...")
    try:
        plot_causal_graph(
            graph,
            channel_names=channels,
            target_name='sales',
            save_path='./causal_graph.png'
        )
        print("   Causal graph saved to ./causal_graph.png")
    except Exception as e:
        print(f"   Visualization skipped: {e}")
    
    print("\n" + "="*60)
    print("Example completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()