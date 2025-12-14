"""
Data loader for MLB batted ball parquet files.
Use this instead of MySQL queries in Colab or any environment.
"""

import pandas as pd
from pathlib import Path

def get_data_path():
    """Find the Data directory relative to this file or repo root."""
    # Check common locations
    possible_paths = [
        Path(__file__).parent.parent / 'Data',  # From Simulator/
        Path('/content/baseball_game_simulator/Data'),  # Colab
        Path('Data'),  # Repo root
    ]
    
    for path in possible_paths:
        if path.exists():
            return path
    
    raise FileNotFoundError("Data directory not found. Ensure repo is cloned.")

def load_batted_balls(seasons=None):
    """
    Load batted ball events.
    
    Args:
        seasons: List of years to load, e.g. [2024, 2025]. None loads all.
    
    Returns:
        DataFrame with all batted ball events.
    """
    data_path = get_data_path() / 'batted_balls'
    
    files = list(data_path.glob('batted_balls_*.parquet'))
    
    if seasons:
        files = [f for f in files if any(str(s) in f.name for s in seasons)]
    
    if not files:
        raise FileNotFoundError(f"No parquet files found in {data_path}")
    
    df = pd.concat([pd.read_parquet(f) for f in sorted(files)], ignore_index=True)
    print(f"Loaded {len(df):,} batted ball events from {len(files)} file(s)")
    
    return df

def load_games():
    """Load games table."""
    path = get_data_path() / 'games' / 'games.parquet'
    df = pd.read_parquet(path)
    print(f"Loaded {len(df):,} games")
    return df

def load_teams():
    """Load teams table."""
    path = get_data_path() / 'teams' / 'teams.parquet'
    df = pd.read_parquet(path)
    print(f"Loaded {len(df):,} teams")
    return df

def load_batted_balls_with_venue(seasons=None):
    """
    Load batted balls joined with venue info from games table.
    This is what Base_Model needs.
    
    Returns:
        DataFrame with batted balls + venue_name column.
    """
    batted_balls = load_batted_balls(seasons)
    games = load_games()
    
    df = batted_balls.merge(
        games[['gamePk', 'venue_name']], 
        on='gamePk', 
        how='left'
    )
    
    # Handle stadium name changes (same as Base_Model.ipynb)
    stadium_mapping = {
        'George M. Steinbrenner Field': 'Yankee Stadium',
        'Sutter Health Park': 'Oakland Coliseum',
        'Daikin Park': 'Minute Maid Park',
        'Rate Field': 'Guaranteed Rate Field'
    }
    df['venue_name'] = df['venue_name'].replace(stadium_mapping)
    
    null_venues = df['venue_name'].isna().sum()
    if null_venues > 0:
        print(f"Warning: {null_venues} rows missing venue_name")
    
    return df
