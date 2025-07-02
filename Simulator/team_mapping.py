"""
Team name mapping utilities for consistent team name handling across the application.
Maps between full team names (from API) and short names (used in constants/visualizations).
"""

# Mapping from full team names (API) to short names (constants)
TEAM_NAME_MAPPING = {
    'Arizona Diamondbacks': 'D-backs',
    'Atlanta Braves': 'Braves',
    'Baltimore Orioles': 'Orioles',
    'Boston Red Sox': 'Red Sox',
    'Chicago White Sox': 'White Sox',
    'Chicago Cubs': 'Cubs',
    'Cincinnati Reds': 'Reds',
    'Cleveland Guardians': 'Guardians',
    'Colorado Rockies': 'Rockies',
    'Detroit Tigers': 'Tigers',
    'Houston Astros': 'Astros',
    'Kansas City Royals': 'Royals',
    'Los Angeles Angels': 'Angels',
    'Los Angeles Dodgers': 'Dodgers',
    'Miami Marlins': 'Marlins',
    'Milwaukee Brewers': 'Brewers',
    'Minnesota Twins': 'Twins',
    'New York Yankees': 'Yankees',
    'New York Mets': 'Mets',
    'Oakland Athletics': 'Athletics',
    'Philadelphia Phillies': 'Phillies',
    'Pittsburgh Pirates': 'Pirates',
    'San Diego Padres': 'Padres',
    'San Francisco Giants': 'Giants',
    'Seattle Mariners': 'Mariners',
    'St. Louis Cardinals': 'Cardinals',
    'Tampa Bay Rays': 'Rays',
    'Texas Rangers': 'Rangers',
    'Toronto Blue Jays': 'Blue Jays',
    'Washington Nationals': 'Nationals'
}

# Reverse mapping for convenience
SHORT_TO_FULL_MAPPING = {v: k for k, v in TEAM_NAME_MAPPING.items()}

# Additional variations that might appear in different contexts
TEAM_VARIATIONS = {
    'LAA': 'Angels',
    'LAD': 'Dodgers',
    'SF': 'Giants', 
    'SD': 'Padres',
    'TB': 'Rays',
    'WSH': 'Nationals',
    'STL': 'Cardinals',
    'CWS': 'White Sox',
    'CHC': 'Cubs',
    'NYY': 'Yankees',
    'NYM': 'Mets',
    'ARI': 'D-backs',
    'ATL': 'Braves',
    'BAL': 'Orioles',
    'BOS': 'Red Sox',
    'CIN': 'Reds',
    'CLE': 'Guardians',
    'COL': 'Rockies',
    'DET': 'Tigers',
    'HOU': 'Astros',
    'KC': 'Royals',
    'MIA': 'Marlins',
    'MIL': 'Brewers',
    'MIN': 'Twins',
    'OAK': 'Athletics',
    'PHI': 'Phillies',
    'PIT': 'Pirates',
    'SEA': 'Mariners',
    'TEX': 'Rangers',
    'TOR': 'Blue Jays'
}


def get_team_short_name(team_name):
    """
    Convert any team name format to the standard short name used in constants.
    
    Args:
        team_name (str): Team name in any format
        
    Returns:
        str: Short team name for use with team_colors dictionary
    """
    if not team_name or pd.isna(team_name):
        return None
        
    # Already a short name
    if team_name in SHORT_TO_FULL_MAPPING:
        return team_name
        
    # Full name to short name
    if team_name in TEAM_NAME_MAPPING:
        return TEAM_NAME_MAPPING[team_name]
        
    # Abbreviation to short name
    if team_name in TEAM_VARIATIONS:
        return TEAM_VARIATIONS[team_name]
        
    # Try partial matching for edge cases
    for full_name, short_name in TEAM_NAME_MAPPING.items():
        if team_name.lower() in full_name.lower() or full_name.lower() in team_name.lower():
            return short_name
            
    # If all else fails, return the original name
    print(f"Warning: Could not map team name '{team_name}' to short name")
    return team_name


def get_team_full_name(team_name):
    """
    Convert any team name format to the full official name.
    
    Args:
        team_name (str): Team name in any format
        
    Returns:
        str: Full official team name
    """
    if not team_name or pd.isna(team_name):
        return None
        
    # Already a full name
    if team_name in TEAM_NAME_MAPPING:
        return team_name
        
    # Short name to full name
    if team_name in SHORT_TO_FULL_MAPPING:
        return SHORT_TO_FULL_MAPPING[team_name]
        
    # Abbreviation to full name
    if team_name in TEAM_VARIATIONS:
        short_name = TEAM_VARIATIONS[team_name]
        return SHORT_TO_FULL_MAPPING.get(short_name, team_name)
        
    return team_name


def get_team_colors_key(team_name):
    """
    Get the correct key for accessing team_colors dictionary.
    
    Args:
        team_name (str): Team name in any format
        
    Returns:
        str: Key for team_colors dictionary
    """
    return get_team_short_name(team_name)


def get_team_logo_filename(team_name):
    """
    Get the correct SVG filename for team logos.
    
    Args:
        team_name (str): Team name in any format
        
    Returns:
        str: SVG filename for team logo
    """
    short_name = get_team_short_name(team_name)
    return f"{short_name}.svg" if short_name else None


# Import pandas for isna check
import pandas as pd
