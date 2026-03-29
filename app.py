import streamlit as st
import pandas as pd
import time
from pipeline import fetch_and_engineer_advanced, update_team_defense, get_top_players_for_team
from model import train_and_predict_advanced
from database import load_from_db

st.set_page_config(page_title="NBA Advanced ML Engine", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #fafafa; }
    .css-1d391kg { background-color: #262730; }
</style>
""", unsafe_allow_html=True)

st.title("NBA Syndicate-Level Predictive Engine")

st.sidebar.header("Mode Selection")
app_mode = st.sidebar.radio("Select View", ["Single Player Analysis", "Full Matchup Projections"])
target_var = st.sidebar.selectbox("Target Variable", ["PTS", "AST", "REB"])

if app_mode == "Single Player Analysis":
    player_name = st.sidebar.text_input("Player Name", "Shai Gilgeous-Alexander")
    
    if st.sidebar.button("Run Advanced Pipeline"):
        with st.spinner("Fetching Dynamic Defense & Training XGBoost..."):
            df, pid = fetch_and_engineer_advanced(player_name)
            
            if df is not None:
                db_df = load_from_db(f'player_{pid}')
                prediction, model, features = train_and_predict_advanced(db_df, target=target_var)
                last_row = db_df.iloc[-1]
                
                st.subheader(f"Projection for {player_name}")
                
                # Updated 5-Column layout to include Deviation
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Projected " + target_var, round(prediction, 1))
                c2.metric("Last Game Actual", last_row[target_var])
                c3.metric("5-Game Avg", round(last_row[f'{target_var}_ROLL_5'], 1))
                c4.metric(f"Volatility (± Deviation)", round(last_row[f'{target_var}_STD_5'], 1))
                c5.metric("Opponent Def Rtg", round(last_row['DEF_RATING'], 1))
                
                st.markdown("---")
                
                # --- AI BETTING INSIGHTS ENGINE ---
                st.markdown("### 💡 AI Betting Insights")
                tips = []
                
                if last_row['MIN_STD_ROLL_5'] > 4.5:
                    tips.append(f"⚠️ **High Rotation Volatility:** This player's minutes are fluctuating heavily. Proceed with caution unless a starter ahead of them is out.")
                if last_row['IS_B2B'] == 1:
                    tips.append("⚠️ **Schedule Fatigue:** Playing on the second night of a back-to-back. High risk of reduced minutes or inefficiency.")
                if last_row['DEF_RATING'] < 111.0:
                    tips.append("🛡️ **Tough Matchup:** The opposing team ranks highly in defensive efficiency. Expect higher resistance.")
                elif last_row['DEF_RATING'] > 115.0:
                    tips.append("🔥 **Favorable Matchup:** The opponent plays weak defense, raising the floor for this prop.")
                if last_row['USG_ROLL_5'] > 28:
                    tips.append("📈 **Alpha Usage:** This player is dominating the team's offensive possessions. Very high floor, especially if a co-star is ruled out tonight.")

                if not tips:
                    tips.append("⚖️ **Neutral Environment:** Standard situational factors. Trust the raw AI projection.")

                for tip in tips:
                    st.info(tip)
                
                st.markdown("---")
                
                col_chart, col_data = st.columns((1, 1))
                with col_chart:
                    st.markdown("### XGBoost Feature Importance")
                    imp_df = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_}).sort_values(by='Importance', ascending=False)
                    st.bar_chart(imp_df.set_index('Feature'))
                with col_data:
                    st.markdown("### Advanced Feature Matrix (Tail)")
                    display_cols = ['GAME_DATE', 'MATCHUP', target_var, f'{target_var}_STD_5', 'DEF_RATING', 'INTERACTION_USG_DEF']
                    st.dataframe(db_df[display_cols].tail(8), height=350)
            else:
                st.error("Player not found or insufficient data.")

elif app_mode == "Full Matchup Projections":
    st.sidebar.markdown("---")
    st.sidebar.header("Matchup Configuration")
    team_a = st.sidebar.text_input("Away Team (Abbreviation)", "LAL")
    team_b = st.sidebar.text_input("Home Team (Abbreviation)", "BOS")
    
    if st.sidebar.button("Generate Matchup Board"):
        st.subheader(f"Projected {target_var} Board: {team_a.upper()} @ {team_b.upper()}")
        
        with st.spinner("Updating League Defensive Ratings..."):
            update_team_defense()
            
        players_a = get_top_players_for_team(team_a)
        players_b = get_top_players_for_team(team_b)
        all_players = players_a + players_b
        
        if not all_players:
            st.error("Could not find rosters. Ensure you use 3-letter abbreviations (e.g., LAL, BOS, NYK).")
        else:
            progress_bar = st.progress(0)
            status_text = st.empty()
            results = []
            
            for i, player in enumerate(all_players):
                status_text.text(f"Processing {player} ({i+1}/{len(all_players)})...")
                df, pid = fetch_and_engineer_advanced(player)
                
                if df is not None:
                    db_df = load_from_db(f'player_{pid}')
                    prediction, model, features = train_and_predict_advanced(db_df, target=target_var)
                    team = team_a if player in players_a else team_b
                    
                    results.append({
                        "Player": player,
                        "Team": team.upper(),
                        f"Projected {target_var}": round(prediction, 1),
                        "5-Game Avg": round(db_df[f'{target_var}_ROLL_5'].iloc[-1], 1),
                        f"Volatility": round(db_df[f'{target_var}_STD_5'].iloc[-1], 1),
                        "Avg Minutes": round(db_df['MIN_ROLL_5'].iloc[-1], 1)
                    })
                
                time.sleep(1) 
                progress_bar.progress((i + 1) / len(all_players))
            
            status_text.text("Matchup Processing Complete!")
            
            if results:
                results_df = pd.DataFrame(results)
                results_df = results_df.sort_values(by=f"Projected {target_var}", ascending=False).reset_index(drop=True)
                st.dataframe(results_df, use_container_width=True)