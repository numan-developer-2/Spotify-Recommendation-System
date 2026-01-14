"""
Model training module for Spotify Recommendation System.
Trains baseline and advanced models for music recommendation.
"""

import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path
from typing import Tuple, Dict
import sys

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from config import get_config
sys.path.append(str(Path(__file__).parent.parent / 'evaluation'))
from metrics import evaluate_classification, evaluate_ranking, print_evaluation_report

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Train and evaluate recommendation models."""
    
    def __init__(self):
        """Initialize model trainer."""
        self.config = get_config()
        self.models_dir = self.config.get_path('paths.models_dir', create_dir=True)
        logger.info(f"ModelTrainer initialized. Models dir: {self.models_dir}")
    
    def load_data(self) -> Tuple[pd.DataFrame, list]:
        """
        Load feature matrix and get feature columns.
        
        Returns:
            Tuple of (features_df, feature_columns)
        """
        logger.info("Loading feature matrix...")
        
        feature_path = self.config.get_path('paths.feature_matrix')
        
        if not feature_path.exists():
            raise FileNotFoundError(f"Feature matrix not found: {feature_path}")
        
        df = pd.read_csv(feature_path)
        logger.info(f"Loaded {len(df):,} examples")
        
        # Define feature columns (exclude ID and label columns)
        exclude_cols = [
            'user_id', 'track_id', 'track_name', 'artist_name', 
            'label', 'similarity_score', 'play_count',
            'playlist_name', 'track_uri', 'artist_uri', 'album_name', 'album_uri',
            'duration_category', 'tempo_category', 'popularity_category', 'energy_level',
            'data_source'
        ]
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        logger.info(f"Using {len(feature_cols)} features")
        
        return df, feature_cols
    
    def prepare_train_test_split(self, df: pd.DataFrame, feature_cols: list,
                                 test_size: float = 0.2) -> Tuple:
        """
        Prepare train-test split.
        
        Args:
            df: Features DataFrame
            feature_cols: List of feature column names
            test_size: Test set proportion
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        logger.info("Preparing train-test split...")
        
        # Separate features and labels
        X = df[feature_cols].fillna(0)  # Fill any remaining NaN
        y = df['label']
        
        # Split data
        random_state = self.config.get('data_processing.random_state', 42)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        logger.info(f"Train set: {len(X_train):,} examples")
        logger.info(f"Test set:  {len(X_test):,} examples")
        logger.info(f"Train positive rate: {y_train.mean():.3f}")
        logger.info(f"Test positive rate:  {y_test.mean():.3f}")
        
        return X_train, X_test, y_train, y_test
    
    def train_baseline_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                            X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """
        Train baseline Logistic Regression model.
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
            
        Returns:
            Dictionary with model and metrics
        """
        logger.info("=" * 60)
        logger.info("Training Baseline Model (Logistic Regression)")
        logger.info("=" * 60)
        
        # Scale features
        logger.info("Scaling features...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        logger.info("Training Logistic Regression...")
        lr_config = self.config.get('model.baseline', {})
        
        model = LogisticRegression(
            solver=lr_config.get('solver', 'liblinear'),
            max_iter=lr_config.get('max_iter', 1000),
            class_weight=lr_config.get('class_weight', 'balanced'),
            random_state=lr_config.get('random_state', 42)
        )
        
        model.fit(X_train_scaled, y_train)
        logger.info("Training complete!")
        
        # Evaluate
        logger.info("Evaluating on test set...")
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # Classification metrics
        class_metrics = evaluate_classification(y_test, y_pred, y_pred_proba)
        
        # Ranking metrics
        rank_metrics = evaluate_ranking(
            y_test.values, y_pred_proba,
            k_values=self.config.get('evaluation.precision_k', [5, 10, 20])
        )
        
        # Combine metrics
        all_metrics = {**class_metrics, **rank_metrics}
        
        # Print report
        print_evaluation_report(all_metrics, "Baseline Model (Logistic Regression)")
        
        # Save model
        model_path = self.models_dir / 'baseline_lr.pkl'
        scaler_path = self.models_dir / 'baseline_scaler.pkl'
        
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        
        logger.info(f"Saved model: {model_path}")
        logger.info(f"Saved scaler: {scaler_path}")
        
        return {
            'model': model,
            'scaler': scaler,
            'metrics': all_metrics,
            'model_path': model_path
        }
    
    def train_lightgbm_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                            X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """
        Train LightGBM model.
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
            
        Returns:
            Dictionary with model and metrics
        """
        logger.info("=" * 60)
        logger.info("Training Advanced Model (LightGBM)")
        logger.info("=" * 60)
        
        # Get LightGBM config
        lgb_config = self.config.get('model.lightgbm', {})
        
        # Create datasets
        logger.info("Creating LightGBM datasets...")
        train_data = lgb.Dataset(X_train, label=y_train)
        test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
        
        # Training parameters
        params = {
            'objective': lgb_config.get('objective', 'binary'),
            'metric': lgb_config.get('metric', 'auc'),
            'boosting_type': lgb_config.get('boosting_type', 'gbdt'),
            'num_leaves': lgb_config.get('num_leaves', 31),
            'learning_rate': lgb_config.get('learning_rate', 0.05),
            'max_depth': lgb_config.get('max_depth', 7),
            'min_child_samples': lgb_config.get('min_child_samples', 20),
            'subsample': lgb_config.get('subsample', 0.8),
            'colsample_bytree': lgb_config.get('colsample_bytree', 0.8),
            'random_state': lgb_config.get('random_state', 42),
            'verbose': lgb_config.get('verbose', -1)
        }
        
        # Train model
        logger.info("Training LightGBM...")
        logger.info(f"Parameters: {params}")
        
        model = lgb.train(
            params,
            train_data,
            num_boost_round=lgb_config.get('n_estimators', 200),
            valid_sets=[train_data, test_data],
            valid_names=['train', 'test'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=20, verbose=False),
                lgb.log_evaluation(period=50)
            ]
        )
        
        logger.info("Training complete!")
        logger.info(f"Best iteration: {model.best_iteration}")
        
        # Evaluate
        logger.info("Evaluating on test set...")
        y_pred_proba = model.predict(X_test, num_iteration=model.best_iteration)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Classification metrics
        class_metrics = evaluate_classification(y_test, y_pred, y_pred_proba)
        
        # Ranking metrics
        rank_metrics = evaluate_ranking(
            y_test.values, y_pred_proba,
            k_values=self.config.get('evaluation.precision_k', [5, 10, 20])
        )
        
        # Combine metrics
        all_metrics = {**class_metrics, **rank_metrics}
        
        # Print report
        print_evaluation_report(all_metrics, "Advanced Model (LightGBM)")
        
        # Feature importance
        logger.info("\nTop 10 Most Important Features:")
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=False)
        
        for idx, row in feature_importance.head(10).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.2f}")
        
        # Save model
        model_path = self.models_dir / 'lightgbm_model.txt'
        model.save_model(str(model_path))
        
        logger.info(f"Saved model: {model_path}")
        
        return {
            'model': model,
            'metrics': all_metrics,
            'feature_importance': feature_importance,
            'model_path': model_path
        }
    
    def run_full_training(self) -> Dict:
        """
        Run complete training pipeline.
        
        Returns:
            Dictionary with all results
        """
        logger.info("=" * 60)
        logger.info("Starting Model Training Pipeline")
        logger.info("=" * 60)
        
        # Step 1: Load data
        logger.info("\n[Step 1/4] Loading data...")
        df, feature_cols = self.load_data()
        
        # Step 2: Train-test split
        logger.info("\n[Step 2/4] Preparing train-test split...")
        test_size = self.config.get('data_processing.test_size', 0.2)
        X_train, X_test, y_train, y_test = self.prepare_train_test_split(
            df, feature_cols, test_size
        )
        
        # Step 3: Train baseline model
        logger.info("\n[Step 3/4] Training baseline model...")
        baseline_results = self.train_baseline_model(X_train, y_train, X_test, y_test)
        
        # Step 4: Train LightGBM model
        logger.info("\n[Step 4/4] Training LightGBM model...")
        lightgbm_results = self.train_lightgbm_model(X_train, y_train, X_test, y_test)
        
        # Compare models
        logger.info("\n" + "=" * 60)
        logger.info("Model Comparison")
        logger.info("=" * 60)
        logger.info(f"Baseline (LR) ROC-AUC:  {baseline_results['metrics']['roc_auc']:.4f}")
        logger.info(f"LightGBM ROC-AUC:       {lightgbm_results['metrics']['roc_auc']:.4f}")
        logger.info(f"Improvement:            {(lightgbm_results['metrics']['roc_auc'] - baseline_results['metrics']['roc_auc']):.4f}")
        
        logger.info("\n" + "=" * 60)
        logger.info("Training Pipeline Complete!")
        logger.info("=" * 60)
        
        return {
            'baseline': baseline_results,
            'lightgbm': lightgbm_results,
            'feature_cols': feature_cols
        }


def main():
    """Train models."""
    trainer = ModelTrainer()
    results = trainer.run_full_training()
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\nModels saved in: {trainer.models_dir}")
    print(f"\nBest model: LightGBM")
    print(f"ROC-AUC: {results['lightgbm']['metrics']['roc_auc']:.4f}")


if __name__ == "__main__":
    main()

