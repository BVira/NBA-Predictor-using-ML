import pandas as pd
import time
from nba_api.stats.endpoints import playergamelog, leaguedashteamstats, teamplayerdashboard
from nba_api.stats.static import players, teams
from database import save_to_db, load_from_db
import sqlite3

def update_team_defense(season="2025-26"):
    team_stats = leaguedashteamstats.LeagueDashTeamStats(season=season, measure_type_detailed_defense='Advanced')
    df = team_stats.get_data_frames()[0]
    df = df[['TEAM_ID', 'TEAM_NAME', 'DEF_RATING', 'PACE']]
    save_to_db(df, 'team_advanced_stats')
    return df

def get_top_players_for_team(team_abbr, season="2025-26", top_n=8):
    """Finds a team's ID and pulls their rotation sorted by MINUTES PLAYED."""
    team_dict = teams.get_teams()
    team_info = [t for t in team_dict if t['abbreviation'] == team_abbr.upper()]
    
    if not team_info:
        return []
    
    team_id = team_info[0]['id']
    
    # Use TeamPlayerDashboard to get actual player stats for the season
    dashboard = teamplayerdashboard.TeamPlayerDashboard(team_id=team_id, season=season)
    df = dashboard.get_data_frames()[1]  # Index 1 contains player-level season totals
    
    # Sort by Minutes (MIN) descending so we get the starters and key bench players
    df = df.sort_values(by='MIN', ascending=False)
    
    return df['PLAYER_NAME'].head(top_n).tolist()

def fetch_and_engineer_advanced(player_name, season="2025-26"):
    player_dict = players.get_players()
    player_info = [p for p in player_dict if p['full_name'].lower() == player_name.lower()]
    
    if not player_info:
        return None, None
    
    pid = player_info[0]['id']
    
    try:
        log = playergamelog.PlayerGameLog(player_id=pid, season=season)
        df = log.get_data_frames()[0]
    except Exception:
        return None, None
    
    if len(df) < 5:
        return None, None
        
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    df = df.sort_values('GAME_DATE').reset_index(drop=True)
    
    # 1. Core Target Rolling Features (Fixes the KeyError)
    for stat in ['PTS', 'AST', 'REB']:
        df[f'{stat}_ROLL_5'] = df[stat].rolling(window=5, min_periods=1).mean()
        df[f'{stat}_STD_5'] = df[stat].rolling(window=5, min_periods=1).std().fillna(0)
    
    df['MIN_ROLL_5'] = df['MIN'].rolling(window=5, min_periods=1).mean()
    df['MIN_STD_ROLL_5'] = df['MIN'].rolling(window=5, min_periods=1).std().fillna(0)
    df['SHOT_ATT_ROLL_5'] = df['FGA'].rolling(window=5, min_periods=1).mean()
    
    # 2. Usage & Efficiency
    df['USG_PROXY'] = df['FGA'] + 0.44 * df['FTA'] + df['TOV']
    df['USG_ROLL_5'] = df['USG_PROXY'].rolling(window=5, min_periods=1).mean()
    df['TS_PCT'] = df['PTS'] / (2 * (df['FGA'] + 0.44 * df['FTA']).replace(0, 1))
    df['TS_ROLL_5'] = df['TS_PCT'].rolling(window=5, min_periods=1).mean()
    
    # 3. Temporal Stress
    df['REST_DAYS'] = df['GAME_DATE'].diff().dt.days.fillna(3)
    df['IS_B2B'] = (df['REST_DAYS'] <= 1).astype(int)
    
    df.set_index('GAME_DATE', inplace=True)
    df['GAMES_LAST_7_DAYS'] = df.index.to_series().diff().dt.days.rolling(window=4, min_periods=1).apply(lambda x: (x <= 7).sum(), raw=False).fillna(1)
    df.reset_index(inplace=True)
    df['HOME_GAME'] = df['MATCHUP'].apply(lambda x: 1 if 'vs.' in x else 0)
    
    # 4. Dynamic Opponent Data
    try:
        team_stats = load_from_db('team_advanced_stats')
    except sqlite3.OperationalError:
        team_stats = update_team_defense(season)
        
    df['OPP_ABBR'] = df['MATCHUP'].str[-3:]
    nba_teams = teams.get_teams()
    abbr_to_name = {t['abbreviation']: t['full_name'] for t in nba_teams}
    df['OPP_NAME'] = df['OPP_ABBR'].map(abbr_to_name)
    
    df = df.merge(team_stats[['TEAM_NAME', 'DEF_RATING', 'PACE']], left_on='OPP_NAME', right_on='TEAM_NAME', how='left')
    df['DEF_RATING'] = df['DEF_RATING'].fillna(115.0)
    df['PACE'] = df['PACE'].fillna(99.0)
    
    # 5. Advanced Interaction Features
    df['INTERACTION_USG_DEF'] = df['USG_PROXY'] * df['DEF_RATING']
    df['INTERACTION_MIN_PACE'] = df['MIN_ROLL_5'] * df['PACE']
    
    df = df.fillna(0)
    save_to_db(df, f'player_{pid}')
    
    return df, pid