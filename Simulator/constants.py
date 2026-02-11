## We want to create variables for the MLB Stats API criteria when pulling game data
base_url = 'https://statsapi.mlb.com/api/'
league = 1 ## League (MLB = 1)
season = 2025
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
# Each stadium is a list of (spray_angle, distance_ft) tuples.
#   spray_angle: degrees from center field (negative=LF, positive=RF)
#   -45 = LF foul pole, 0 = dead center, +45 = RF foul pole
#
# ~10-15 waypoints per park trace the actual fence shape.
# Linear interpolation between points avoids cubic overshoot.
# =============================================================================

STADIUM_DIMENSIONS = {
    # ----- American League East -----

    'Oriole Park at Camden Yards': [  # 2025 dimensions
        (-45, 333),
        (-40, 337),
        (-35, 346),
        (-30, 357),
        (-22.5, 364),
        (-15, 378),
        (-7, 393),
        (0, 400),
        (7, 393),
        (15, 382),
        (22.5, 373),
        (30, 358),
        (35, 340),
        (40, 326),
        (45, 318),
    ],

    'Fenway Park': [  # Green Monster + deep triangle + short Pesky Pole
        (-45, 310),   # LF foul pole (Green Monster)
        (-40, 310),   # Monster continues flat
        (-36, 310),   # Monster continues flat
        (-34, 310),   # Monster end — sharp corner
        (-32, 335),   # Wall steps back abruptly
        (-28, 360),   # Transition to deep left-center
        (-22, 379),   # Left-center
        (-15, 400),   # Approaching the triangle
        (-10, 420),   # Deep triangle apex
        (-5, 408),    # Coming back from triangle
        (0, 390),     # Center field
        (7, 388),     # Right-center
        (15, 383),    # Right-center
        (22.5, 380),  # Right-center
        (30, 355),    # Approaching bullpen area
        (35, 338),    # Right field
        (40, 318),    # Near Pesky Pole
        (45, 302),    # RF foul pole (Pesky Pole)
    ],

    'Yankee Stadium': [  # Short porch in RF
        (-45, 318),
        (-40, 325),
        (-35, 342),
        (-30, 365),
        (-22.5, 399),
        (-15, 405),
        (-7, 408),
        (0, 408),
        (7, 403),
        (15, 395),
        (22.5, 385),
        (30, 363),
        (35, 342),
        (40, 325),
        (45, 314),
    ],

    'Tropicana Field': [
        (-45, 315),
        (-40, 322),
        (-35, 335),
        (-30, 350),
        (-22.5, 370),
        (-15, 385),
        (-7, 398),
        (0, 404),
        (7, 398),
        (15, 385),
        (22.5, 370),
        (30, 352),
        (35, 338),
        (40, 328),
        (45, 322),
    ],

    'Rogers Centre': [  # Post-2023 renovation
        (-45, 328),
        (-40, 332),
        (-35, 342),
        (-30, 355),
        (-22.5, 368),
        (-15, 380),
        (-7, 393),
        (0, 400),
        (7, 390),
        (15, 375),
        (22.5, 359),
        (30, 347),
        (35, 338),
        (40, 332),
        (45, 328),
    ],

    # ----- American League Central -----

    'Guaranteed Rate Field': [
        (-45, 330),
        (-40, 335),
        (-35, 348),
        (-30, 362),
        (-22.5, 377),
        (-15, 386),
        (-7, 395),
        (0, 400),
        (7, 393),
        (15, 383),
        (22.5, 372),
        (30, 358),
        (35, 347),
        (40, 340),
        (45, 335),
    ],

    'Progressive Field': [
        (-45, 325),
        (-40, 330),
        (-35, 342),
        (-30, 356),
        (-22.5, 370),
        (-15, 383),
        (-7, 398),
        (0, 405),
        (7, 398),
        (15, 388),
        (22.5, 375),
        (30, 358),
        (35, 342),
        (40, 332),
        (45, 325),
    ],

    'Comerica Park': [  # Post-2023 (CF was 420, now 412)
        (-45, 342),
        (-40, 345),
        (-35, 352),
        (-30, 360),
        (-22.5, 370),
        (-15, 385),
        (-7, 400),
        (0, 412),
        (7, 398),
        (15, 382),
        (22.5, 365),
        (30, 352),
        (35, 340),
        (40, 334),
        (45, 330),
    ],

    'Kauffman Stadium': [  # Symmetrical
        (-45, 330),
        (-40, 338),
        (-35, 352),
        (-30, 368),
        (-22.5, 387),
        (-15, 397),
        (-7, 406),
        (0, 410),
        (7, 406),
        (15, 397),
        (22.5, 387),
        (30, 368),
        (35, 352),
        (40, 338),
        (45, 330),
    ],

    'Target Field': [
        (-45, 339),
        (-40, 342),
        (-35, 352),
        (-30, 365),
        (-22.5, 377),
        (-15, 388),
        (-7, 398),
        (0, 404),
        (7, 396),
        (15, 382),
        (22.5, 367),
        (30, 352),
        (35, 340),
        (40, 332),
        (45, 328),
    ],

    # ----- American League West -----

    'Minute Maid Park': [  # Tal's Hill removed; short LF, deep CF
        (-45, 315),
        (-40, 318),
        (-35, 328),
        (-30, 342),
        (-22.5, 362),
        (-15, 380),
        (-7, 398),
        (0, 409),
        (7, 400),
        (15, 388),
        (22.5, 373),
        (30, 356),
        (35, 342),
        (40, 332),
        (45, 326),
    ],

    'Angel Stadium': [
        (-45, 330),
        (-40, 338),
        (-35, 352),
        (-30, 368),
        (-22.5, 387),
        (-15, 393),
        (-7, 396),
        (0, 396),
        (7, 393),
        (15, 385),
        (22.5, 370),
        (30, 356),
        (35, 344),
        (40, 336),
        (45, 330),
    ],

    'Oakland Coliseum': [  # Symmetrical
        (-45, 330),
        (-40, 338),
        (-35, 352),
        (-30, 370),
        (-22.5, 388),
        (-15, 395),
        (-7, 398),
        (0, 400),
        (7, 398),
        (15, 395),
        (22.5, 388),
        (30, 370),
        (35, 352),
        (40, 338),
        (45, 330),
    ],

    'T-Mobile Park': [
        (-45, 331),
        (-40, 335),
        (-35, 346),
        (-30, 360),
        (-22.5, 378),
        (-15, 389),
        (-7, 397),
        (0, 401),
        (7, 398),
        (15, 390),
        (22.5, 381),
        (30, 362),
        (35, 346),
        (40, 334),
        (45, 326),
    ],

    'Globe Life Field': [
        (-45, 329),
        (-40, 334),
        (-35, 346),
        (-30, 358),
        (-22.5, 372),
        (-15, 385),
        (-7, 398),
        (0, 407),
        (7, 398),
        (15, 387),
        (22.5, 374),
        (30, 356),
        (35, 342),
        (40, 332),
        (45, 326),
    ],

    # Temporary/alternate AL venues
    'Daikin Park': [  # → same as Minute Maid Park
        (-45, 315),
        (-40, 318),
        (-35, 328),
        (-30, 342),
        (-22.5, 362),
        (-15, 380),
        (-7, 398),
        (0, 409),
        (7, 400),
        (15, 388),
        (22.5, 373),
        (30, 356),
        (35, 342),
        (40, 332),
        (45, 326),
    ],

    'Rate Field': [  # → same as Guaranteed Rate Field
        (-45, 330),
        (-40, 335),
        (-35, 348),
        (-30, 362),
        (-22.5, 377),
        (-15, 386),
        (-7, 395),
        (0, 400),
        (7, 393),
        (15, 383),
        (22.5, 372),
        (30, 358),
        (35, 347),
        (40, 340),
        (45, 335),
    ],

    'Sutter Health Park': [  # → same as Oakland Coliseum
        (-45, 330),
        (-40, 338),
        (-35, 352),
        (-30, 370),
        (-22.5, 388),
        (-15, 395),
        (-7, 398),
        (0, 400),
        (7, 398),
        (15, 395),
        (22.5, 388),
        (30, 370),
        (35, 352),
        (40, 338),
        (45, 330),
    ],

    'George M. Steinbrenner Field': [  # → same as Yankee Stadium
        (-45, 318),
        (-40, 325),
        (-35, 342),
        (-30, 365),
        (-22.5, 399),
        (-15, 405),
        (-7, 408),
        (0, 408),
        (7, 403),
        (15, 395),
        (22.5, 385),
        (30, 363),
        (35, 342),
        (40, 325),
        (45, 314),
    ],

    # ----- National League East -----

    'Citi Field': [  # Post-renovations
        (-45, 335),
        (-40, 338),
        (-35, 344),
        (-30, 350),
        (-22.5, 358),
        (-15, 375),
        (-7, 395),
        (0, 408),
        (7, 400),
        (15, 392),
        (22.5, 380),
        (30, 362),
        (35, 348),
        (40, 338),
        (45, 330),
    ],

    'Citizens Bank Park': [  # "The Angle" notch in left-center
        (-45, 329),
        (-40, 335),
        (-35, 348),
        (-30, 362),
        (-22.5, 374),
        (-18, 390),
        (-15, 409),   # The Angle — deep notch
        (-10, 408),
        (-5, 404),
        (0, 401),
        (7, 393),
        (15, 381),
        (22.5, 369),
        (30, 354),
        (35, 342),
        (40, 335),
        (45, 330),
    ],

    'Nationals Park': [
        (-45, 336),
        (-40, 340),
        (-35, 350),
        (-30, 363),
        (-22.5, 377),
        (-15, 387),
        (-7, 397),
        (0, 402),
        (7, 393),
        (15, 382),
        (22.5, 370),
        (30, 356),
        (35, 346),
        (40, 340),
        (45, 335),
    ],

    'Truist Park': [
        (-45, 335),
        (-40, 342),
        (-35, 355),
        (-30, 370),
        (-22.5, 385),
        (-15, 393),
        (-7, 398),
        (0, 400),
        (7, 395),
        (15, 387),
        (22.5, 375),
        (30, 358),
        (35, 342),
        (40, 332),
        (45, 325),
    ],

    'loanDepot park': [  # Post-2020 (CF was 407)
        (-45, 344),
        (-40, 348),
        (-35, 358),
        (-30, 370),
        (-22.5, 384),
        (-15, 392),
        (-7, 397),
        (0, 400),
        (7, 398),
        (15, 394),
        (22.5, 387),
        (30, 370),
        (35, 355),
        (40, 344),
        (45, 335),
    ],

    # ----- National League Central -----

    'Wrigley Field': [  # Roughly symmetrical alleys
        (-45, 355),
        (-40, 355),
        (-35, 358),
        (-30, 363),
        (-22.5, 368),
        (-15, 378),
        (-7, 392),
        (0, 400),
        (7, 392),
        (15, 378),
        (22.5, 368),
        (30, 363),
        (35, 358),
        (40, 355),
        (45, 353),
    ],

    'Great American Ball Park': [
        (-45, 328),
        (-40, 335),
        (-35, 350),
        (-30, 365),
        (-22.5, 379),
        (-15, 390),
        (-7, 400),
        (0, 404),
        (7, 396),
        (15, 383),
        (22.5, 370),
        (30, 354),
        (35, 340),
        (40, 330),
        (45, 325),
    ],

    'American Family Field': [
        (-45, 344),
        (-40, 346),
        (-35, 352),
        (-30, 360),
        (-22.5, 370),
        (-15, 380),
        (-7, 392),
        (0, 400),
        (7, 394),
        (15, 385),
        (22.5, 374),
        (30, 364),
        (35, 355),
        (40, 349),
        (45, 345),
    ],

    'PNC Park': [  # Deep left-center notch at ~410
        (-45, 325),
        (-40, 332),
        (-35, 348),
        (-30, 368),
        (-22.5, 389),
        (-18, 400),
        (-15, 410),   # Deep left-center nook
        (-10, 408),
        (-5, 402),
        (0, 399),
        (7, 393),
        (15, 383),
        (22.5, 375),
        (30, 356),
        (35, 340),
        (40, 328),
        (45, 320),
    ],

    'Busch Stadium': [  # Symmetrical
        (-45, 336),
        (-40, 340),
        (-35, 350),
        (-30, 362),
        (-22.5, 375),
        (-15, 385),
        (-7, 395),
        (0, 400),
        (7, 395),
        (15, 385),
        (22.5, 375),
        (30, 362),
        (35, 350),
        (40, 340),
        (45, 335),
    ],

    # ----- National League West -----

    'Chase Field': [
        (-45, 330),
        (-40, 336),
        (-35, 348),
        (-30, 360),
        (-22.5, 374),
        (-15, 387),
        (-7, 400),
        (0, 407),
        (7, 400),
        (15, 387),
        (22.5, 374),
        (30, 360),
        (35, 348),
        (40, 340),
        (45, 334),
    ],

    'Coors Field': [  # Deepest CF in MLB
        (-45, 347),
        (-40, 352),
        (-35, 362),
        (-30, 375),
        (-22.5, 390),
        (-15, 402),
        (-7, 412),
        (0, 415),
        (7, 405),
        (15, 390),
        (22.5, 375),
        (30, 365),
        (35, 357),
        (40, 353),
        (45, 350),
    ],

    'Dodger Stadium': [  # Symmetrical
        (-45, 330),
        (-40, 338),
        (-35, 352),
        (-30, 368),
        (-22.5, 385),
        (-15, 391),
        (-7, 394),
        (0, 395),
        (7, 394),
        (15, 391),
        (22.5, 385),
        (30, 368),
        (35, 352),
        (40, 338),
        (45, 330),
    ],

    'Oracle Park': [  # Triples Alley in RC, short RF to McCovey Cove
        (-45, 339),
        (-40, 342),
        (-35, 350),
        (-30, 358),
        (-22.5, 364),
        (-15, 374),
        (-7, 385),
        (0, 391),
        (5, 395),
        (10, 402),
        (15, 410),
        (20, 415),    # Triples Alley — deepest point
        (25, 410),
        (30, 399),
        (35, 370),
        (40, 338),
        (45, 309),    # Short RF porch to McCovey Cove
    ],

    'Petco Park': [
        (-45, 336),
        (-40, 342),
        (-35, 356),
        (-30, 372),
        (-22.5, 386),
        (-15, 392),
        (-7, 395),
        (0, 396),
        (7, 395),
        (15, 393),
        (22.5, 391),
        (30, 370),
        (35, 350),
        (40, 334),
        (45, 322),
    ],
}

DEFAULT_STADIUM_DIMENSIONS = [
    (-45, 330), (-35, 352), (-22.5, 375), (-15, 385), (-7, 395),
    (0, 400), (7, 395), (15, 385), (22.5, 375), (35, 352), (45, 330),
]

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
