"""
Main orchestrator for Spotify Recommendation System.
Runs the complete end-to-end pipeline.
"""

import logging
import argparse
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent))

from config import get_config
from src.data_loader import DataLoader
from src.preprocessing import DataPreprocessor
from src.labeling import LabelGenerator
from features.build_features import FeatureBuilder
from src.train import ModelTrainer
from src.recommend import RecommendationEngine

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_data_pipeline(n_users: int = 1000, interactions_per_user: int = 50):
    """
    Run data loading, preprocessing, and labeling pipeline.
    
    Args:
        n_users: Number of synthetic users to create
        interactions_per_user: Average interactions per user
    """
    logger.info("=" * 70)
    logger.info("PHASE 1: DATA PIPELINE")
    logger.info("=" * 70)
    
    # Step 1: Preprocessing
    logger.info("\n[1/3] Running preprocessing...")
    preprocessor = DataPreprocessor()
    audio_df, _ = preprocessor.run_full_preprocessing(max_playlists=1000)
    
    # Step 2: Label generation
    logger.info("\n[2/3] Generating labels...")
    generator = LabelGenerator()
    labels_df = generator.run_full_labeling(
        audio_df,
        n_users=n_users,
        interactions_per_user=interactions_per_user
    )
    
    # Step 3: Feature engineering
    logger.info("\n[3/3] Building features...")
    builder = FeatureBuilder()
    features_df, feature_cols = builder.run_full_feature_engineering(audio_df, labels_df)
    
    logger.info("\n" + "=" * 70)
    logger.info("DATA PIPELINE COMPLETE!")
    logger.info("=" * 70)
    
    return features_df, feature_cols


def run_training_pipeline():
    """Run model training pipeline."""
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 2: MODEL TRAINING")
    logger.info("=" * 70)
    
    trainer = ModelTrainer()
    results = trainer.run_full_training()
    
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING PIPELINE COMPLETE!")
    logger.info("=" * 70)
    
    return results


def run_recommendation_demo(n_users: int = 5):
    """
    Run recommendation demo.
    
    Args:
        n_users: Number of users to generate recommendations for
    """
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 3: RECOMMENDATION DEMO")
    logger.info("=" * 70)
    
    # Initialize recommendation engine
    engine = RecommendationEngine(model_type='lightgbm')
    
    # Generate recommendations for sample users
    for user_id in range(1, n_users + 1):
        print("\n" + "=" * 70)
        print(f"🎵 Recommendations for User {user_id}")
        print("=" * 70)
        
        recommendations = engine.recommend(user_id, top_k=10)
        
        print(f"\nTop 10 Recommended Songs:\n")
        for idx, row in recommendations.iterrows():
            print(f"{idx + 1:2d}. {row['track_name'][:50]:<50} | {row['artist_name'][:30]:<30}")
            print(f"     Score: {row['score']:.4f} | Pop: {row['popularity']:3d} | "
                  f"Dance: {row['danceability']:.2f} | Energy: {row['energy']:.2f}")
    
    logger.info("\n" + "=" * 70)
    logger.info("RECOMMENDATION DEMO COMPLETE!")
    logger.info("=" * 70)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Spotify Recommendation System')
    parser.add_argument(
        '--mode',
        type=str,
        choices=['full', 'data', 'train', 'recommend'],
        default='full',
        help='Pipeline mode to run'
    )
    parser.add_argument(
        '--n-users',
        type=int,
        default=1000,
        help='Number of synthetic users to create'
    )
    parser.add_argument(
        '--interactions',
        type=int,
        default=50,
        help='Average interactions per user'
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 70)
    logger.info("🎧 SPOTIFY RECOMMENDATION SYSTEM")
    logger.info("=" * 70)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Users: {args.n_users}")
    logger.info(f"Interactions per user: {args.interactions}")
    logger.info("=" * 70)
    
    try:
        if args.mode == 'full':
            # Run complete pipeline
            logger.info("\nRunning FULL PIPELINE...\n")
            
            # Phase 1: Data
            run_data_pipeline(args.n_users, args.interactions)
            
            # Phase 2: Training
            run_training_pipeline()
            
            # Phase 3: Demo
            run_recommendation_demo(n_users=5)
            
        elif args.mode == 'data':
            # Run only data pipeline
            run_data_pipeline(args.n_users, args.interactions)
            
        elif args.mode == 'train':
            # Run only training
            run_training_pipeline()
            
        elif args.mode == 'recommend':
            # Run only recommendation demo
            run_recommendation_demo(n_users=10)
        
        logger.info("\n" + "=" * 70)
        logger.info("✅ ALL PIPELINES COMPLETED SUCCESSFULLY!")
        logger.info("=" * 70)
        
        # Print summary
        config = get_config()
        print("\n" + "=" * 70)
        print("📊 PROJECT SUMMARY")
        print("=" * 70)
        print(f"\nProject Root: {config.project_root}")
        print(f"\nProcessed Data:")
        print(f"  - Audio Features: {config.get_path('paths.audio_features_combined')}")
        print(f"  - Labels: {config.get_path('paths.labels')}")
        print(f"  - Features: {config.get_path('paths.feature_matrix')}")
        print(f"\nTrained Models:")
        print(f"  - Baseline (LR): {config.get_path('paths.models_dir') / 'baseline_lr.pkl'}")
        print(f"  - LightGBM: {config.get_path('paths.models_dir') / 'lightgbm_model.txt'}")
        print("\n" + "=" * 70)
        
    except Exception as e:
        logger.error(f"\n❌ Pipeline failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

