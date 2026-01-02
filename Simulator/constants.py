## We want to create variables for the MLB Stats API criteria when pulling game data
base_url = 'https://statsapi.mlb.com/api/'
league = 1 ## League (MLB = 1)
season = 2026
endpoint = f'schedule?language=en&sportId={league}&season={season}'

## Team version
team_ver = 'v1'
## Schedule version
schedule_ver = 'v1'
## Game version
game_ver = 'v1.1'

## Venue names to filter for the model, since we don't have batted ball data on games played at different parks
venue_names = ['Oriole Park at Camden Yards', 'Great American Ball Park',
       'Petco Park', 'Dodger Stadium', 'Tropicana Field',
       'Kauffman Stadium', 'Rate Field', 'loanDepot park',
       'Daikin Park', 'Globe Life Field', 'Sutter Health Park',
       'Chase Field', 'T-Mobile Park', 'Citi Field', 'Citizens Bank Park',
       'Wrigley Field', 'Nationals Park', 'American Family Field',
       'Target Field', 'Busch Stadium', 'Yankee Stadium', 'Comerica Park',
       'Coors Field', 'PNC Park', 'Oracle Park', 'Truist Park',
       'Angel Stadium', 'Progressive Field', 'Rogers Centre',
       'Fenway Park', 'George M. Steinbrenner Field']

## Colors and Logo
team_colors = {
    "D-backs": ("#A71930", "Process Black"),
    "Braves": ("#CE1141", "108"),
    "Orioles": ("#DF4601", "109"),
    "Red Sox": ("#BD3039", "116"),
    "White Sox": ("#27251F", "122"),
    "Cubs": ("#0E3386", "123"),
    "Reds": ("#C6011F", "124"),
    "Guardians": ("#00385D", "1235"),
    "Rockies": ("#333366", "155"),
    "Tigers": ("#0C2C56", "1655"),
    "Astros": ("#EB6E1F", "172"),
    "Royals": ("#004687", "Warm Red"),
    "Angels": ("#BA0021", "1815"),
    "Dodgers": ("#005A9C", "185"),
    "Marlins": ("#00A3E0", "186"),
    "Brewers": ("#FFC52f", "187"),
    "Twins": ("#002B5C", "199"),
    "Yankees": ("#003087", "200"),
    "Mets": ("#002D72", "201"),
    "Athletics": ("#003831", "202"),
    "Phillies": ("#E81828", "273"),
    "Pirates": ("#FDB827", "2767"),
    "Padres": ("#2F241D", "282"),
    "Giants": ("#FD5A1E", "287"),
    "Mariners": ("#0C2C56", "288"),
    "Cardinals": ("#C41E3A", "289"),
    "Rays": ("#092C5C", "292"),
    "Rangers": ("#C0111F", "293"),
    "Blue Jays": ("#134A8E", "294"),
    "Nationals": ("#AB0003", "295"),
    "National League All-Stars": ("#87CEEB", "blue"),
    "American League All-Stars": ("#FF7F50", "coral")
}

# Dictionary of MLB teams and their corresponding ESPN logo URLs
mlb_team_logos = [
    {"team": "D-backs", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/ari.png&h=500&w=500"},
    {"team": "Braves", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/atl.png&h=500&w=500"},
    {"team": "Orioles", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/bal.png&h=500&w=500"},
    {"team": "Red Sox", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/bos.png&h=500&w=500"},
    {"team": "Cubs", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/chc.png&h=500&w=500"},
    {"team": "White Sox", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/chw.png&h=500&w=500"},
    {"team": "Reds", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/cin.png&h=500&w=500"},
    {"team": "Guardians", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/cle.png&h=500&w=500"},
    {"team": "Rockies", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/col.png&h=500&w=500"},
    {"team": "Tigers", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/det.png&h=500&w=500"},
    {"team": "Astros", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/hou.png&h=500&w=500"},
    {"team": "Royals", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/kc.png&h=500&w=500"},
    {"team": "Angels", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/laa.png&h=500&w=500"},
    {"team": "Dodgers", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/lad.png&h=500&w=500"},
    {"team": "Marlins", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/mia.png&h=500&w=500"},
    {"team": "Brewers", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/mil.png&h=500&w=500"},
    {"team": "Twins", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/min.png&h=500&w=500"},
    {"team": "Mets", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/nym.png&h=500&w=500"},
    {"team": "Yankees", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/nyy.png&h=500&w=500"},
    {"team": "Athletics", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/oak.png&h=500&w=500"},
    {"team": "Phillies", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/phi.png&h=500&w=500"},
    {"team": "Pirates", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/pit.png&h=500&w=500"},
    {"team": "Padres", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/sd.png&h=500&w=500"},
    {"team": "Giants", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/sf.png&h=500&w=500"},
    {"team": "Mariners", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/sea.png&h=500&w=500"},
    {"team": "Cardinals", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/stl.png&h=500&w=500"},
    {"team": "Rays", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/tb.png&h=500&w=500"},
    {"team": "Rangers", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/tex.png&h=500&w=500"},
    {"team": "Blue Jays", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/tor.png&h=500&w=500"},
    {"team": "Nationals", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/wsh.png&h=500&w=500"},
       {"team": "American League All-Stars", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/al.png&h=500&w=500"},
       {"team": "National League All-Stars", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/nl.png&h=500&w=500"}
]

# =============================================================================
# VENUE ID MAPPING
# =============================================================================
# Maps venue names to venue IDs for the batted ball model.
# venue_id is more stable than venue_name (names can change year to year).

VENUE_NAME_TO_ID = {
    'Oriole Park at Camden Yards': '2',
    'Fenway Park': '3',
    'Yankee Stadium': '3313',
    'Tropicana Field': '12',
    'Rogers Centre': '14',
    'Chase Field': '15',
    'Coors Field': '19',
    'Dodger Stadium': '22',
    'Kauffman Stadium': '7',
    'Angel Stadium': '1',
    'Oakland Coliseum': '10',
    'Petco Park': '2680',
    'Oracle Park': '2395',
    'T-Mobile Park': '680',
    'Globe Life Field': '5325',
    'Minute Maid Park': '2392',
    'Target Field': '3312',
    'Guaranteed Rate Field': '4',
    'Comerica Park': '2394',
    'Progressive Field': '5',
    'PNC Park': '31',
    'Great American Ball Park': '2602',
    'Busch Stadium': '2889',
    'Wrigley Field': '17',
    'American Family Field': '32',
    'Citi Field': '3289',
    'Citizens Bank Park': '2681',
    'Nationals Park': '3309',
    'Truist Park': '4705',
    'loanDepot park': '4169',
    # Temporary/alternate venues (map to similar parks)
    'George M. Steinbrenner Field': '3313',  # → Yankee Stadium
    'Sutter Health Park': '10',               # → Oakland Coliseum
    'Daikin Park': '2392',                    # → Minute Maid Park
    'Rate Field': '4',                        # → Guaranteed Rate Field
}

# Default venue ID for unknown stadiums
DEFAULT_VENUE_ID = '22'

# =============================================================================
# STADIUM DIMENSIONS (in feet)
# =============================================================================
# Standard keys: LF, LCF, CF, RCF, RF (at angles -45, -22.5, 0, 22.5, 45)
# Optional extra keys for quirky parks:
#   DLCF = Deep Left-Center Field (angle ~ -15)
#   DRCF = Deep Right-Center Field (angle ~ +30)
#   LF_CORNER, RF_CORNER = additional corner points
# =============================================================================

STADIUM_DIMENSIONS = {
    # ----- American League East -----
    'Oriole Park at Camden Yards': {'LF': 333, 'LCF': 364, 'CF': 400, 'RCF': 373, 'RF': 318},  # 2025 dimensions
    'Fenway Park':                 {'LF': 310, 'LCF': 379, 'DLCF': 420, 'CF': 390, 'RCF': 380, 'RF': 302},  # Triangle at 420
    'Yankee Stadium':              {'LF': 318, 'LCF': 399, 'CF': 408, 'RCF': 385, 'RF': 314},
    'Tropicana Field':             {'LF': 315, 'LCF': 370, 'CF': 404, 'RCF': 370, 'RF': 322},  # Rays home (not Steinbrenner)
    'Rogers Centre':               {'LF': 328, 'LCF': 368, 'CF': 400, 'RCF': 359, 'RF': 328},  # Post-2023 renovation
    
    # ----- American League Central -----
    'Guaranteed Rate Field':       {'LF': 330, 'LCF': 377, 'CF': 400, 'RCF': 372, 'RF': 335},
    'Progressive Field':           {'LF': 325, 'LCF': 370, 'CF': 405, 'RCF': 375, 'RF': 325},
    'Comerica Park':               {'LF': 342, 'LCF': 370, 'CF': 412, 'RCF': 365, 'RF': 330},  # Post-2023 (CF was 420)
    'Kauffman Stadium':            {'LF': 330, 'LCF': 387, 'CF': 410, 'RCF': 387, 'RF': 330},  # Symmetrical
    'Target Field':                {'LF': 339, 'LCF': 377, 'CF': 404, 'RCF': 367, 'RF': 328},
    
    # ----- American League West -----
    'Minute Maid Park':            {'LF': 315, 'LCF': 362, 'CF': 409, 'RCF': 373, 'RF': 326},  # Tal's Hill removed
    'Angel Stadium':               {'LF': 330, 'LCF': 387, 'CF': 396, 'RCF': 370, 'RF': 330},
    'Oakland Coliseum':            {'LF': 330, 'LCF': 388, 'CF': 400, 'RCF': 388, 'RF': 330},  # Symmetrical
    'T-Mobile Park':               {'LF': 331, 'LCF': 378, 'CF': 401, 'RCF': 381, 'RF': 326},
    'Globe Life Field':            {'LF': 329, 'LCF': 372, 'CF': 407, 'RCF': 374, 'RF': 326},
    
    # ----- National League East -----
    'Citi Field':                  {'LF': 335, 'LCF': 358, 'CF': 408, 'RCF': 380, 'RF': 330},  # Post-renovations
    'Citizens Bank Park':          {'LF': 329, 'LCF': 374, 'DLCF': 409, 'CF': 401, 'RCF': 369, 'RF': 330},  # "The Angle" at 409
    'Nationals Park':              {'LF': 336, 'LCF': 377, 'CF': 402, 'RCF': 370, 'RF': 335},
    'Truist Park':                 {'LF': 335, 'LCF': 385, 'CF': 400, 'RCF': 375, 'RF': 325},
    'loanDepot park':              {'LF': 344, 'LCF': 384, 'CF': 400, 'RCF': 387, 'RF': 335},  # Post-2020 (CF was 407)
    
    # ----- National League Central -----
    'Wrigley Field':               {'LF': 355, 'LCF': 368, 'CF': 400, 'RCF': 368, 'RF': 353},  # Symmetrical alleys
    'Great American Ball Park':    {'LF': 328, 'LCF': 379, 'CF': 404, 'RCF': 370, 'RF': 325},
    'American Family Field':       {'LF': 344, 'LCF': 370, 'CF': 400, 'RCF': 374, 'RF': 345},
    'PNC Park':                    {'LF': 325, 'LCF': 389, 'DLCF': 410, 'CF': 399, 'RCF': 375, 'RF': 320},  # Deep nook at 410
    'Busch Stadium':               {'LF': 336, 'LCF': 375, 'CF': 400, 'RCF': 375, 'RF': 335},  # Symmetrical
    
    # ----- National League West -----
    'Chase Field':                 {'LF': 330, 'LCF': 374, 'CF': 407, 'RCF': 374, 'RF': 334},
    'Coors Field':                 {'LF': 347, 'LCF': 390, 'CF': 415, 'RCF': 375, 'RF': 350},  # Deepest CF in MLB
    'Dodger Stadium':              {'LF': 330, 'LCF': 385, 'CF': 395, 'RCF': 385, 'RF': 330},  # Symmetrical
    'Oracle Park':                 {'LF': 339, 'LCF': 364, 'CF': 391, 'DRCF': 415, 'RCF': 399, 'RF': 309},  # Triples Alley at 415
    'Petco Park':                  {'LF': 336, 'LCF': 386, 'CF': 396, 'RCF': 391, 'RF': 322},
}

DEFAULT_STADIUM_DIMENSIONS = {'LF': 330, 'LCF': 375, 'CF': 400, 'RCF': 375, 'RF': 330}

DEFAULT_STADIUM_DIMENSIONS = {'LF': 330, 'LCF': 375, 'CF': 400, 'RCF': 375, 'RF': 330}

# Conversion factor: feet to plot units
FEET_TO_PLOT = 100 / 400

# =============================================================================
# TEAM NAME MAPPINGS
# =============================================================================

# Short name -> logo lookup name
TEAM_LOGO_MAP = {
    'Jays': 'Blue Jays',
    'Sox': 'Red Sox',
    'D-backs': 'D-backs',
}

# Short name -> full display name  
TEAM_DISPLAY_MAP = {
    'Jays': 'Blue Jays',
    'Sox': 'Red Sox',
    'D-backs': 'D-backs',
}
