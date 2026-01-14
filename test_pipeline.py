"""
Quick test script to verify the complete pipeline works.
Runs a minimal version of the full pipeline for testing.
"""

import sys
from pathlib import Path

# Add to path
sys.path.append(str(Path(__file__).parent))

from config import get_config
from src.preprocessing import DataPreprocessor
from src.labeling import LabelGenerator
from features.build_features import FeatureBuilder

print("=" * 70)
print("QUICK TEST - Spotify Recommendation System")
print("=" * 70)

try:
    # Test 1: Config
    print("\n[Test 1/4] Testing configuration...")
    config = get_config()
    print(f"[OK] Config loaded. Project root: {config.project_root}")
    
    # Test 2: Preprocessing (small sample)
    print("\n[Test 2/4] Testing preprocessing...")
    preprocessor = DataPreprocessor()
    audio_df, _ = preprocessor.run_full_preprocessing(max_playlists=100)
    print(f"[OK] Preprocessing complete. Loaded {len(audio_df):,} tracks")
    
    # Test 3: Label generation (small sample)
    print("\n[Test 3/4] Testing label generation...")
    generator = LabelGenerator()
    labels_df = generator.run_full_labeling(
        audio_df,
        n_users=100,  # Small for testing
        interactions_per_user=30
    )
    print(f"[OK] Labels generated. Created {len(labels_df):,} examples")
    
    # Test 4: Feature engineering
    print("\n[Test 4/4] Testing feature engineering...")
    builder = FeatureBuilder()
    features_df, feature_cols = builder.run_full_feature_engineering(audio_df, labels_df)
    print(f"[OK] Features built. {len(features_df):,} examples, {len(feature_cols)} features")
    
    print("\n" + "=" * 70)
    print("[SUCCESS] ALL TESTS PASSED!")
    print("=" * 70)
    print("\nYou can now run the full pipeline with:")
    print("  python main.py --mode full --n-users 1000 --interactions 50")
    print("\nOr train models with existing data:")
    print("  python main.py --mode train")
    print("=" * 70)
    
except Exception as e:
    print(f"\n[ERROR] Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
