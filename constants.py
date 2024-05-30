## We want to create variables for the MLB Stats API criteria when pulling game data
base_url = 'https://statsapi.mlb.com/api/'
league = 1 ## League (MLB = 1)
season = 2024
endpoint = f'schedule?language=en&sportId={league}&season={season}'

## Schedule version
schedule_ver = 'v1'

## Game version
game_ver = 'v1.1'