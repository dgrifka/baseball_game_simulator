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
