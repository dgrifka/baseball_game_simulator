## We want to create variables for the MLB Stats API criteria when pulling game data
base_url = 'https://statsapi.mlb.com/api/'
league = 1 ## League (MLB = 1)
season = 2024
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
       'Kauffman Stadium', 'Guaranteed Rate Field', 'loanDepot park',
       'Minute Maid Park', 'Globe Life Field', 'Oakland Coliseum',
       'Chase Field', 'T-Mobile Park', 'Citi Field', 'Citizens Bank Park',
       'Wrigley Field', 'Nationals Park', 'American Family Field',
       'Target Field', 'Busch Stadium', 'Yankee Stadium', 'Comerica Park',
       'Coors Field', 'PNC Park', 'Oracle Park', 'Truist Park',
       'Angel Stadium', 'Progressive Field', 'Rogers Centre',
       'Fenway Park']

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
    "Nationals": ("#AB0003", "295")
}
