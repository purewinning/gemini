import streamlit as st
import pandas as pd
from pulp import *
import plotly.express as px
import numpy as np
import warnings

# Suppress SettingWithCopyWarning from Pandas, which often occurs during heuristic assignment (like FLEX)
warnings.filterwarnings('ignore')

# --- 1. DATA LOADING AND PREPARATION ---
@st.cache_data
def load_and_prepare_data(file_name):
    """Loads the CSV, renames columns, and calculates value."""
    try:
        df = pd.read_csv(file_name)
    except FileNotFoundError:
        st.error(f"File not found: {file_name}. Please ensure the file is in the correct directory.")
        return pd.DataFrame() 

    # Rename 'Projection' to 'PFP' (Projected Fantasy Points) for consistent code
    if 'Projection' in df.columns:
        df = df.rename(columns={'Projection': 'PFP'})
    else:
        st.error("The 'Projection' column (PFP) was not found in the file.")
        return pd.DataFrame()

    # Calculate 'Value' (PFP per $1000 salary) and handle division by zero
    df['Value'] = df.apply(lambda row: row['PFP'] / (row['Salary'] / 1000) if row['Salary'] > 0 else 0, axis=1)

    # Filter out players with zero salary or projection (likely inactive/unavailable)
    df_filtered = df[(df['Salary'] > 0) & (df['PFP'] >= 0)].copy()

    # Select only the columns needed for the optimizer and visualization
    df_final = df_filtered[['Player', 'Salary', 'Position', 'Team', 'PFP', 'Value']].reset_index(drop=True)
    return df_final

file_name = "NFL_DK_Main_Projections.csv (1).csv"
df = load_and_prepare_data(file_name)
df_initial = df.copy() # Keep a clean copy for resets

# Exit if data loading failed
if df.empty:
    st.stop()

# --- 2. STREAMLIT APP SETUP ---
st.set_page_config(layout="wide", page_title="Ultimate Fantasy Optimizer")

st.title("🏆 The Ultimate AI Fantasy Optimizer")
st.caption("Using your uploaded projections to generate a Portfolio of Optimized Lineups.")

# --- SIDEBAR CONTROLS & FILTERS ---
with st.sidebar:
    st.header("1. Core Settings")

    # Global Settings (DraftKings NFL Classic)
    MAX_SALARY = 50000
    CAP_LIMIT = st.slider("Max Salary Cap Limit:", min_value=45000, max_value=MAX_SALARY, value=MAX_SALARY, step=500)
    NUM_LINEUPS = st.slider("Number of Unique Lineups to Generate:", min_value=1, max_value=20, value=10, step=1)

    st.header("2. Player Lock & Exclude")
    st.info("Lock a player to force them into ALL generated lineups. Exclude a player to ensure they are never used.")

    # Player Selection Widgets
    all_players = df['Player'].tolist()
    locked_players = st.multiselect("🔒 Players to LOCK:", all_players)
    excluded_players = st.multiselect("❌ Players to EXCLUDE:", all_players, default=[]) # Empty default for clean slate

# --- 3. THE OPTIMIZATION CORE (PULP) ---

def run_optimization(df, cap, num_lineups, locked_players, excluded_players):
    """Generates multiple unique, optimized lineups."""

    all_lineups = []

    # Filter the main DataFrame based on user exclusions
    df_filtered = df[~df['Player'].isin(excluded_players)].reset_index(drop=True)

    # Dictionary to hold the constraints for uniqueness across lineups
    previous_lineup_constraints = {}

    for k in range(num_lineups):
        prob = LpProblem(f"Lineup_Optimization_{k+1}", LpMaximize)
        player_vars = LpVariable.dicts("Select", df_filtered.index, 0, 1, LpBinary)

        # 1. OBJECTIVE FUNCTION: Maximize total PFP
        prob += lpSum([df_filtered['PFP'][i] * player_vars[i] for i in df_filtered.index]), "Total Projected Fantasy Points"

        # 2. CONSTRAINTS (Standard DFS Rules for NFL DK Classic)
        pos_map = df_filtered.groupby('Position').groups

        # QB (Exactly 1)
        prob += lpSum([player_vars[i] for i in pos_map.get('QB', [])]) == 1, "QB Constraint"
        # RB (Min 2)
        prob += lpSum([player_vars[i] for i in pos_map.get('RB', [])]) >= 2, "Min RB Constraint"
        # WR (Min 3)
        prob += lpSum([player_vars[i] for i in pos_map.get('WR', [])]) >= 3, "Min WR Constraint"
        # TE (Min 1)
        prob += lpSum([player_vars[i] for i in pos_map.get('TE', [])]) >= 1, "Min TE Constraint"
        # DEF (Exactly 1)
        prob += lpSum([player_vars[i] for i in pos_map.get('DEF', [])]) == 1, "DEF Constraint"
        # Total Players (Exactly 9)
        prob += lpSum([player_vars[i] for i in df_filtered.index]) == 9, "Total Players Constraint"
        # Salary Cap Constraint
        prob += lpSum([df_filtered['Salary'][i] * player_vars[i] for i in df_filtered.index]) <= cap, "Salary Cap Constraint"

        # 3. PLAYER LOCK CONSTRAINTS
        for player_name in locked_players:
            lock_index = df_filtered[df_filtered['Player'] == player_name].index
            if not lock_index.empty:
                 prob += player_vars[lock_index[0]] == 1, f"Lock {player_name}"

        # 4. UNIQUENESS CONSTRAINT (Enforce diversity from previous solutions)
        if k > 0 and previous_lineup_constraints:
            # Enforce at least 3 players be different from the previous optimal lineup
            # Sum of (1 - player_var[i]) for the previous solution where i was selected must be >= 3
            prob += lpSum(player_vars[i] for i in df_filtered.index) - lpSum(previous_lineup_constraints[i] for i in df_filtered.index) <= 6, f"Unique_Constraint_{k}"


        # Solve the problem
        prob.solve(PULP_CBC_CMD(msg=0)) 

        if prob.status == LpStatusOptimal:
            # Extract the solution
            selected_players_indices = [i for i in df_filtered.index if player_vars[i].varValue == 1]
            lineup_df = df_filtered.iloc[selected_players_indices].copy()

            # Assign Flex position for clarity (Heuristic: the player who fills the extra RB/WR/TE spot)
            current_roster = lineup_df['Position'].value_counts()
            
            # Identify the extra spot to label as FLEX
            flex_candidates = []
            if current_roster.get('RB', 0) > 2: flex_candidates.extend(lineup_df[lineup_df['Position'] == 'RB'].nlargest(current_roster.get('RB', 0) - 2, 'PFP').index.tolist())
            if current_roster.get('WR', 0) > 3: flex_candidates.extend(lineup_df[lineup_df['Position'] == 'WR'].nlargest(current_roster.get('WR', 0) - 3, 'PFP').index.tolist())
            if current_roster.get('TE', 0) > 1: flex_candidates.extend(lineup_df[lineup_df['Position'] == 'TE'].nlargest(current_roster.get('TE', 0) - 1, 'PFP').index.tolist())

            if flex_candidates:
                # Find the player with the lowest PFP among the candidates and label them FLEX
                flex_player_index = lineup_df.loc[flex_candidates].nsmallest(1, 'PFP').index[0]
                lineup_df.loc[flex_player_index, 'Position'] = 'FLEX'
                
            lineup_df['Lineup #'] = k + 1
            all_lineups.append(lineup_df)

            # Update the constraint set for the next iteration
            previous_lineup_constraints = {i: player_vars[i].varValue for i in df_filtered.index}
        else:
            st.warning(f"Solver failed to find optimal Lineup #{k + 1}. Status: {LpStatus[prob.status]}. Stopping portfolio generation.")
            break

    return pd.concat(all_lineups) if all_lineups else None

# --- 4. DISPLAY RESULTS ---

st.header("1. Optimize Lineup Portfolio")
if st.button(f"Generate Portfolio of {NUM_LINEUPS} Optimized Lineups"):

    with st.spinner(f'Generating {NUM_LINEUPS} unique lineups...'):
        all_lineups_df = run_optimization(df_initial, CAP_LIMIT, NUM_LINEUPS, locked_players, excluded_players)

    if all_lineups_df is not None:

        st.subheader("📊 Portfolio Analysis")

        col_metrics, col_summary = st.columns([1, 2])

        # --- Player Exposure Chart ---
        exposure_data = all_lineups_df.groupby('Player').size().reset_index(name='Exposure Count')
        exposure_data['Exposure %'] = (exposure_data['Exposure Count'] / NUM_LINEUPS) * 100
        exposure_data = exposure_data.sort_values('Exposure %', ascending=False)
        active_exposure = exposure_data[exposure_data['Exposure %'] > 0]

        with col_metrics:
            st.metric("Total Lineups Generated", NUM_LINEUPS)
            st.metric("Average PFP", f"{all_lineups_df.groupby('Lineup #')['PFP'].sum().mean():.2f}")
            if not active_exposure.empty:
                st.metric("Top Player Exposure", active_exposure.iloc[0]['Player'])
            else:
                 st.metric("Top Player Exposure", "N/A")


        with col_summary:
            st.markdown("### Top Player Exposure")
            # Display player exposure in a bar chart
            fig_exposure = px.bar(
                exposure_data.head(10),
                x='Player',
                y='Exposure %',
                color='Exposure %',
                title='Top 10 Player Exposure Across Portfolio',
                template='plotly_white'
            )
            fig_exposure.update_layout(xaxis={'categoryorder':'total descending'}, yaxis_range=[0, 100])
            st.plotly_chart(fig_exposure, use_container_width=True)

        st.subheader("Optimized Lineups Table (Best PFP View)")

        # Display the best lineup (Lineup 1)
        best_lineup = all_lineups_df[all_lineups_df['Lineup #'] == 1].drop(columns=['Lineup #'])
        total_pfp = best_lineup['PFP'].sum()
        total_salary = best_lineup['Salary'].sum()

        col_pfp, col_salary = st.columns(2)
        with col_pfp:
             st.success(f"**Lineup 1 PFP:** {total_pfp:.2f} Points")
        with col_salary:
             st.info(f"**Lineup 1 Salary:** ${total_salary:,.0f}")

        # Re-order the lineup by position for clean display
        pos_order = ['QB', 'RB', 'WR', 'TE', 'FLEX', 'DEF']
        best_lineup['Position'] = pd.Categorical(best_lineup['Position'], categories=pos_order, ordered=True)
        best_lineup = best_lineup.sort_values('Position')

        st.dataframe(
            best_lineup[['Position', 'Player', 'Team', 'Salary', 'PFP', 'Value']],
            use_container_width=True,
            hide_index=True
        )

        with st.expander("View Full Portfolio Details (Raw Data)"):
            st.dataframe(all_lineups_df.drop(columns=['Team']))


# --- 5. INTERACTIVE VALUE PICKER ---

st.header("2. Player Value & Filtering")
st.markdown("Interactively filter players to target or fade based on their value score.")

col_selector, col_chart = st.columns([1, 2])

# Player filtering based on Value
with col_selector:
    max_value = df_initial['Value'].max()
    min_value = st.slider("Minimum Value Score (PFP/Salary in thousands):",
                          min_value=df_initial['Value'].min(),
                          max_value=max_value,
                          value=df_initial['Value'].quantile(0.85), # Start high
                          step=0.1)

    filtered_df = df_initial[df_initial['Value'] >= min_value]
    st.subheader(f"Players with Value ≥ {min_value:.1f}:")
    st.dataframe(filtered_df[['Player', 'Position', 'Value']].sort_values('Value', ascending=False), use_container_width=True, hide_index=True)


with col_chart:
    # Scatter plot with interactive Value threshold
    fig = px.scatter(
        df_initial,
        x='Salary',
        y='PFP',
        color='Position',
        size='Value',
        hover_name='Player',
        template='plotly_white',
        title='Salary vs. Projected Fantasy Points (PFP)',
        labels={'Salary': 'Player Salary ($)', 'PFP': 'Projected Fantasy Points'},
        color_discrete_map={'QB': 'red', 'RB': 'blue', 'WR': 'green', 'TE': 'orange', 'FLEX': 'gray', 'DEF': 'purple'}
    )

    # Add a visual line to represent the user's Value threshold
    max_salary = df_initial['Salary'].max()
    x_range = [0, max_salary]
    y_threshold = [min_value * (x_range[0] / 1000), min_value * (x_range[1] / 1000)]

    fig.add_scatter(
        x=x_range,
        y=y_threshold,
        mode='lines',
        line=dict(color='Red', width=3, dash='dash'),
        name=f'Value Threshold ({min_value:.1f})'
    )
    fig.update_xaxes(range=[0, max_salary * 1.05])
    fig.update_yaxes(range=[0, df_initial['PFP'].max() * 1.05])

    st.plotly_chart(fig, use_container_width=True)
