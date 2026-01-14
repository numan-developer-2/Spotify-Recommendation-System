"""
Data loader module for Spotify Recommendation System.
Loads raw datasets: audio features CSVs and playlist JSON.
"""

import json
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from config import get_config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataLoader:
    """Load and validate raw datasets."""
    
    def __init__(self):
        """Initialize data loader with configuration."""
        self.config = get_config()
        self.project_root = self.config.project_root
        logger.info(f"DataLoader initialized. Project root: {self.project_root}")
    
    def load_audio_features(self, dataset: str = 'both') -> pd.DataFrame:
        """
        Load Spotify audio features datasets.
        
        Args:
            dataset: Which dataset to load ('april', 'nov', or 'both')
            
        Returns:
            DataFrame with audio features
        """
        logger.info(f"Loading audio features dataset: {dataset}")
        
        if dataset == 'april':
            path = self.config.get_path('paths.audio_features_april')
            df = self._load_csv(path, 'April 2019')
            return df
        
        elif dataset == 'nov':
            path = self.config.get_path('paths.audio_features_nov')
            df = self._load_csv(path, 'November 2018')
            return df
        
        elif dataset == 'both':
            # Load both datasets
            april_path = self.config.get_path('paths.audio_features_april')
            nov_path = self.config.get_path('paths.audio_features_nov')
            
            df_april = self._load_csv(april_path, 'April 2019')
            df_nov = self._load_csv(nov_path, 'November 2018')
            
            # Add source column to track origin
            df_april['data_source'] = 'april_2019'
            df_nov['data_source'] = 'nov_2018'
            
            # Combine datasets
            logger.info("Combining both datasets...")
            df_combined = pd.concat([df_april, df_nov], ignore_index=True)
            
            # Remove duplicates based on track_id (keep most recent = April)
            initial_count = len(df_combined)
            df_combined = df_combined.drop_duplicates(subset=['track_id'], keep='first')
            duplicates_removed = initial_count - len(df_combined)
            
            logger.info(f"Removed {duplicates_removed:,} duplicate tracks")
            logger.info(f"Final combined dataset: {len(df_combined):,} tracks")
            
            return df_combined
        
        else:
            raise ValueError(f"Invalid dataset: {dataset}. Choose 'april', 'nov', or 'both'")
    
    def _load_csv(self, path: Path, name: str) -> pd.DataFrame:
        """
        Load CSV file with error handling.
        
        Args:
            path: Path to CSV file
            name: Dataset name for logging
            
        Returns:
            DataFrame
        """
        if not path.exists():
            raise FileNotFoundError(f"{name} dataset not found at: {path}")
        
        logger.info(f"Loading {name} from: {path}")
        df = pd.read_csv(path)
        logger.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
        
        # Log basic info
        logger.info(f"Columns: {list(df.columns)}")
        missing = df.isnull().sum().sum()
        logger.info(f"Total missing values: {missing}")
        
        return df
    
    def load_playlists(self, max_playlists: int = None) -> Dict:
        """
        Load Spotify Million Playlist Dataset challenge set.
        
        Args:
            max_playlists: Maximum number of playlists to load (None = all)
            
        Returns:
            Dictionary with playlist data
        """
        path = self.config.get_path('paths.playlist_challenge')
        
        if not path.exists():
            raise FileNotFoundError(f"Playlist dataset not found at: {path}")
        
        logger.info(f"Loading playlist dataset from: {path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"Dataset version: {data.get('version', 'unknown')}")
        logger.info(f"Dataset date: {data.get('date', 'unknown')}")
        
        playlists = data.get('playlists', [])
        total_playlists = len(playlists)
        logger.info(f"Total playlists available: {total_playlists:,}")
        
        # Limit playlists if requested
        if max_playlists and max_playlists < total_playlists:
            playlists = playlists[:max_playlists]
            logger.info(f"Limited to {max_playlists:,} playlists")
        
        # Calculate statistics
        total_tracks = sum(len(p.get('tracks', [])) for p in playlists)
        avg_tracks = total_tracks / len(playlists) if playlists else 0
        
        logger.info(f"Total tracks across playlists: {total_tracks:,}")
        logger.info(f"Average tracks per playlist: {avg_tracks:.1f}")
        
        return {
            'version': data.get('version'),
            'date': data.get('date'),
            'playlists': playlists,
            'metadata': {
                'total_playlists': len(playlists),
                'total_tracks': total_tracks,
                'avg_tracks_per_playlist': avg_tracks
            }
        }
    
    def load_all_datasets(self, max_playlists: int = None) -> Tuple[pd.DataFrame, Dict]:
        """
        Load all datasets at once.
        
        Args:
            max_playlists: Maximum playlists to load (None = all)
            
        Returns:
            Tuple of (audio_features_df, playlist_data)
        """
        logger.info("=" * 60)
        logger.info("Loading all datasets...")
        logger.info("=" * 60)
        
        # Load audio features
        audio_features = self.load_audio_features('both')
        
        logger.info("")
        
        # Load playlists
        playlists = self.load_playlists(max_playlists)
        
        logger.info("=" * 60)
        logger.info("All datasets loaded successfully!")
        logger.info("=" * 60)
        
        return audio_features, playlists
    
    def get_dataset_summary(self) -> Dict:
        """
        Get summary statistics of all datasets without loading full data.
        
        Returns:
            Dictionary with dataset summaries
        """
        summary = {}
        
        # Audio features summary
        try:
            april_path = self.config.get_path('paths.audio_features_april')
            nov_path = self.config.get_path('paths.audio_features_nov')
            
            df_april = pd.read_csv(april_path, nrows=0)
            df_nov = pd.read_csv(nov_path, nrows=0)
            
            summary['audio_features'] = {
                'april_columns': list(df_april.columns),
                'nov_columns': list(df_nov.columns),
                'columns_match': list(df_april.columns) == list(df_nov.columns)
            }
        except Exception as e:
            summary['audio_features'] = {'error': str(e)}
        
        # Playlist summary
        try:
            playlist_path = self.config.get_path('paths.playlist_challenge')
            with open(playlist_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            summary['playlists'] = {
                'version': data.get('version'),
                'date': data.get('date'),
                'total_playlists': len(data.get('playlists', []))
            }
        except Exception as e:
            summary['playlists'] = {'error': str(e)}
        
        return summary


def main():
    """Test data loading functionality."""
    loader = DataLoader()
    
    # Test 1: Load audio features
    print("\n" + "=" * 60)
    print("TEST 1: Loading Audio Features")
    print("=" * 60)
    audio_df = loader.load_audio_features('both')
    print(f"\nShape: {audio_df.shape}")
    print(f"\nFirst 3 rows:")
    print(audio_df.head(3))
    print(f"\nData types:")
    print(audio_df.dtypes)
    
    # Test 2: Load playlists (limited to 100 for testing)
    print("\n" + "=" * 60)
    print("TEST 2: Loading Playlists (first 100)")
    print("=" * 60)
    playlist_data = loader.load_playlists(max_playlists=100)
    print(f"\nMetadata: {playlist_data['metadata']}")
    print(f"\nFirst playlist sample:")
    print(f"  PID: {playlist_data['playlists'][0]['pid']}")
    print(f"  Name: {playlist_data['playlists'][0].get('name', 'N/A')}")
    print(f"  Tracks: {playlist_data['playlists'][0]['num_tracks']}")
    
    # Test 3: Dataset summary
    print("\n" + "=" * 60)
    print("TEST 3: Dataset Summary")
    print("=" * 60)
    summary = loader.get_dataset_summary()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

