"""
MLB Batted Ball Data Export Script
Exports MySQL data to Parquet format for the baseball_data repo.

Usage: Run in VS Code when connected to local MySQL after Database_Update.
Typically run twice per year (mid-season and end-of-season).
"""

import pandas as pd
import os
from sqlalchemy import create_engine

# ============================================================
# CONFIGURATION
# ============================================================

# MySQL connection
DB_USERNAME = 'root'
DB_PASSWORD = 'password'
DB_HOST = 'localhost'
DB_NAME = 'mlb'

# Output location
OUTPUT_DIR = '/Users/derekgrifka/Desktop/Victory_Analytics/baseball_data'

# Seasons to export (update as needed)
SEASONS = [2024, 2025]

# Batted ball columns to keep
BATTED_BALL_COLUMNS = [
    'playId', 'ab_num', 'gamePk', 'season',
    'inning', 'isTopInning', 'outs', 'balls', 'strikes',
    'awayScore', 'homeScore',
    'batter_id', 'pitcher_id', 'batSide_code', 'pitchHand_code',
    'eventType', 'isOut', 'isScoringPlay', 'rbi',
    'hitData_launchSpeed', 'hitData_launchAngle',
    'hitData_coordinates_coordX', 'hitData_coordinates_coordY',
    'hitData_trajectory', 'hitData_hardness',
    'hitData_location', 'hitData_totalDistance'
]

# ============================================================
# EXPORT FUNCTIONS
# ============================================================

def create_engine_connection():
    connection_string = f'mysql+pymysql://{DB_USERNAME}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}'
    return create_engine(connection_string)

def export_games(engine):
    """Export games table, excluding Spring Training & Exhibition."""
    print("\n" + "="*50)
    print("EXPORTING GAMES")
    print("="*50)
    
    seasons_str = ', '.join(map(str, SEASONS))
    query = f"""
    SELECT * FROM games 
    WHERE seriesDescription NOT IN ('Spring Training', 'Exhibition')
      AND season IN ({seasons_str})
    """
    games_df = pd.read_sql(query, engine)
    valid_game_pks = set(games_df['gamePk'].unique())
    
    os.makedirs(f'{OUTPUT_DIR}/data/games', exist_ok=True)
    games_path = f'{OUTPUT_DIR}/data/games/games.parquet'
    games_df.to_parquet(games_path, compression='snappy', index=False)
    
    print(f"Seasons: {sorted(games_df['season'].unique())}")
    print(f"Rows: {len(games_df):,}")
    print(f"Unique gamePks: {len(valid_game_pks):,}")
    print(f"File: {os.path.getsize(games_path) / (1024*1024):.2f} MB")
    
    return valid_game_pks

def export_batted_balls(engine, valid_game_pks):
    """Export batted ball events for each season."""
    print("\n" + "="*50)
    print("EXPORTING BATTED BALLS")
    print("="*50)
    
    os.makedirs(f'{OUTPUT_DIR}/data/batted_balls', exist_ok=True)
    sql_columns = ', '.join(BATTED_BALL_COLUMNS)
    
    for season in SEASONS:
        print(f"\n{season}:")
        
        query = f"""
        SELECT {sql_columns}
        FROM game_info 
        WHERE season = {season} 
          AND hitData_launchSpeed IS NOT NULL
        """
        df = pd.read_sql(query, engine)
        
        # Filter to valid games & deduplicate
        before_filter = len(df)
        df = df[df['gamePk'].isin(valid_game_pks)]
        after_filter = len(df)
        df = df.drop_duplicates(subset='playId', keep='last')
        after_dedup = len(df)
        
        # Save
        output_path = f'{OUTPUT_DIR}/data/batted_balls/batted_balls_{season}.parquet'
        df.to_parquet(output_path, compression='snappy', index=False)
        
        print(f"  Raw: {before_filter:,} → Filtered: {after_filter:,} → Deduped: {after_dedup:,}")
        print(f"  File: {os.path.getsize(output_path) / (1024*1024):.2f} MB")

def export_teams(engine):
    """Export teams table."""
    print("\n" + "="*50)
    print("EXPORTING TEAMS")
    print("="*50)
    
    os.makedirs(f'{OUTPUT_DIR}/data/teams', exist_ok=True)
    teams_df = pd.read_sql("SELECT * FROM teams", engine)
    teams_path = f'{OUTPUT_DIR}/data/teams/teams.parquet'
    teams_df.to_parquet(teams_path, compression='snappy', index=False)
    
    print(f"Rows: {len(teams_df):,}")

def cleanup_old_files():
    """Remove parquet files for seasons not in SEASONS list."""
    batted_balls_dir = f'{OUTPUT_DIR}/data/batted_balls'
    if os.path.exists(batted_balls_dir):
        for file in os.listdir(batted_balls_dir):
            if file.endswith('.parquet'):
                # Extract year from filename
                try:
                    year = int(file.replace('batted_balls_', '').replace('.parquet', ''))
                    if year not in SEASONS:
                        os.remove(os.path.join(batted_balls_dir, file))
                        print(f"Removed old file: {file}")
                except ValueError:
                    pass

def verify_export():
    """Verify exported data integrity."""
    print("\n" + "="*50)
    print("VERIFICATION")
    print("="*50)
    
    # Load all data
    batted_balls = pd.concat([
        pd.read_parquet(f'{OUTPUT_DIR}/data/batted_balls/batted_balls_{s}.parquet')
        for s in SEASONS
    ], ignore_index=True)
    games = pd.read_parquet(f'{OUTPUT_DIR}/data/games/games.parquet')
    
    # Check joins
    bb_pks = set(batted_balls['gamePk'].unique())
    games_pks = set(games['gamePk'].unique())
    unmatched = bb_pks - games_pks
    
    print(f"Total batted balls: {len(batted_balls):,}")
    print(f"Total games: {len(games):,}")
    print(f"Unmatched gamePks: {len(unmatched)}")
    
    # File summary
    print("\nFiles:")
    total_size = 0
    for root, dirs, files in os.walk(f'{OUTPUT_DIR}/data'):
        for file in files:
            if file.endswith('.parquet'):
                path = os.path.join(root, file)
                size = os.path.getsize(path) / (1024 * 1024)
                total_size += size
                print(f"  {file}: {size:.2f} MB")
    print(f"Total: {total_size:.2f} MB")
    
    return len(unmatched) == 0

# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    engine = create_engine_connection()
    
    valid_game_pks = export_games(engine)
    export_batted_balls(engine, valid_game_pks)
    export_teams(engine)
    cleanup_old_files()
    
    if verify_export():
        print("\n✓ Export complete — ready to commit to GitHub!")
    else:
        print("\n⚠️ Verification failed — check for issues before committing")
