import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
import re

# Load and train the model
@st.cache_data
def load_model():
    # Load the CSV data
    matches = pd.read_csv("matches2.csv", index_col=0)

    # Add the home column (1 for home, 0 for away)
    matches['home'] = 1  # Assume all 'team' entries are the home team
    
    # Prepare the data
    X = matches[['team', 'opponent', 'home']]
    y_result = matches['result']  # Match outcome (win/lose/draw)
    y_team_score = matches['gf']  # Team's score
    y_opponent_score = matches['ga'] # Opponent's score
    y_team_shots = matches['sh']  # Shots
    y_team_sot = matches['sot']  # Shots on target
    y_team_xg = matches['xg']  # Expected goals
    y_team_xa = matches['xga']  # Expected assists

    # Convert team names into dummy variables
    X = pd.get_dummies(X, columns=['team', 'opponent'])
    
    # Split data for result, scores, and shots
    X_train, X_test, y_train_result, y_test_result = train_test_split(X, y_result, test_size=0.2, random_state=1)
    y_train_team_score, y_test_team_score = train_test_split(y_team_score, test_size=0.2, random_state=1)
    y_train_opponent_score, y_test_opponent_score = train_test_split(y_opponent_score, test_size=0.2, random_state=1)
    y_train_team_shots, y_test_team_shots = train_test_split(y_team_shots, test_size=0.2, random_state=1)
    y_train_team_sot, y_test_team_sot = train_test_split(y_team_sot, test_size=0.2, random_state=1)
    y_train_team_xg, y_test_team_xg = train_test_split(y_team_xg, test_size=0.2, random_state=1)
    y_train_team_xa, y_test_team_xa = train_test_split(y_team_xa, test_size=0.2, random_state=1)

    # Train the models
    result_model = RandomForestClassifier(n_estimators=1000, random_state=1)
    result_model.fit(X_train, y_train_result)
    
    team_score_model = RandomForestRegressor(n_estimators=1000, random_state=1)
    team_score_model.fit(X_train, y_train_team_score)
    
    opponent_score_model = RandomForestRegressor(n_estimators=1000, random_state=1)
    opponent_score_model.fit(X_train, y_train_opponent_score)
    
    team_shots_model = RandomForestRegressor(n_estimators=1000, random_state=1)
    team_shots_model.fit(X_train, y_train_team_shots)
    
    team_sot_model = RandomForestRegressor(n_estimators=1000, random_state=1)
    team_sot_model.fit(X_train, y_train_team_sot)

    team_xg_model = RandomForestRegressor(n_estimators=1000, random_state=1)
    team_xg_model.fit(X_train, y_train_team_xg)

    team_xa_model = RandomForestRegressor(n_estimators=1000, random_state=1)
    team_xa_model.fit(X_train, y_train_team_xa)

    # Predict on the test set
    y_pred_result = result_model.predict(X_test)
    y_pred_team_score = team_score_model.predict(X_test)
    y_pred_opponent_score = opponent_score_model.predict(X_test)
    y_pred_team_shots = team_shots_model.predict(X_test)
    y_pred_team_sot = team_sot_model.predict(X_test)
    
    # Calculate accuracy for result and errors for score predictions
    result_accuracy = accuracy_score(y_test_result, y_pred_result)
    team_score_error = mean_squared_error(y_test_team_score, y_pred_team_score, squared=False)
    opponent_score_error = mean_squared_error(y_test_opponent_score, y_pred_opponent_score, squared=False)
    team_shots_error = mean_squared_error(y_test_team_shots, y_pred_team_shots, squared=False)
    team_sot_error = mean_squared_error(y_test_team_sot, y_pred_team_sot, squared=False)
    
    # Return the models, dataset, and metrics
    return result_model, team_score_model, opponent_score_model, team_shots_model, team_sot_model, team_xg_model, team_xa_model, matches, result_accuracy, team_score_error, opponent_score_error, team_shots_error, team_sot_error

# Predict the outcome and scores of a match between two teams
def predict_match(team1, team2, result_model, team_score_model, opponent_score_model, team_shots_model, team_sot_model, team_xg_model, team_xa_model, matches, home=1):
    # Check if both teams exist in the dataset
    if team1 not in matches['team'].unique() or team2 not in matches['team'].unique():
        return f"One or both of the teams '{team1}' and '{team2}' do not exist in the dataset."

    # Prepare the input for prediction (convert team names into a suitable format for the model)
    input_data = pd.DataFrame([[team1, team2, home]], columns=['team', 'opponent', 'home'])
    input_data = pd.get_dummies(input_data)

    # Ensure the columns match those of the training data
    all_columns = pd.get_dummies(matches[['team', 'opponent', 'home']]).columns
    input_data = input_data.reindex(columns=all_columns, fill_value=0)

    # Make predictions
    predicted_result = result_model.predict(input_data)[0]
    predicted_team_score = round(team_score_model.predict(input_data)[0], 1)
    predicted_opponent_score = round(opponent_score_model.predict(input_data)[0], 1)
    predicted_team_shots = round(team_shots_model.predict(input_data)[0], 1)
    predicted_team_sot = round(team_sot_model.predict(input_data)[0], 1)
    predicted_team_xg = round(team_xg_model.predict(input_data)[0], 1)
    predicted_team_xa = round(team_xa_model.predict(input_data)[0], 1)

    # Add a home advantage bias to the result
    home_advantage_factor = 0.4  # This could be tweaked based on domain knowledge
    if home == 1 and predicted_team_score < predicted_opponent_score:
        predicted_team_score += home_advantage_factor

    # Return the predicted result, scores, and shots on target
    return (predicted_result, predicted_team_score, predicted_opponent_score, 
            predicted_team_shots, predicted_team_sot, predicted_team_xg, predicted_team_xa)

# Parse user input question like 'Who will win if Liverpool vs Arsenal?'
def parse_question(question):
    # Extract team names by looking for the pattern "Team1 vs Team2"
    match = re.search(r'([a-zA-Z\s]+) vs ([a-zA-Z\s]+)\??', question, re.IGNORECASE)
    
    if match:
        team1 = match.group(1).strip()
        team2 = match.group(2).strip()
        return team1, team2
    return None, None

# Streamlit UI components
st.title("Football Match Predictor ⚽")
st.write("Ask me about a match and I'll predict the outcome and stats!")

# Load the model and data
result_model, team_score_model, opponent_score_model, team_shots_model, team_sot_model, team_xg_model, team_xa_model, matches, result_accuracy, team_score_error, opponent_score_error, team_shots_error, team_sot_error = load_model()

st.write(f"Model accuracy for result prediction: {result_accuracy:.2f}")
st.write(f"Team score RMSE: {team_score_error:.2f}")
st.write(f"Opponent score RMSE: {opponent_score_error:.2f}")
st.write(f"Team shots RMSE: {team_shots_error:.2f}")
st.write(f"Team shots on target RMSE: {team_sot_error:.2f}")

# Create a form with text input and a button
with st.form(key="match_predict_form"):
    question = st.text_input("Ask your question (e.g., 'Liverpool vs Arsenal?')")
    submit_button = st.form_submit_button(label="Predict")

if submit_button and question:
    team1, team2 = parse_question(question)

    if team1 and team2:
        result, team_score, opponent_score, team_shots, team_sot, team_xg, team_xa = predict_match(
            team1, team2, result_model, team_score_model, opponent_score_model, team_shots_model, 
            team_sot_model, team_xg_model, team_xa_model, matches
        )
        
        st.write(f"Prediction: {result}")
        st.write(f"Score: {team1} {team_score} - {team2} {opponent_score}")
        st.write(f"Shots: {team1} {team_shots}")
        st.write(f"Shots on target: {team1} {team_sot}")
        st.write(f"Expected goals (xG): {team1} {team_xg}")
        st.write(f"Expected assists (xA): {team1} {team_xa}")
    else:
        st.write("Could not understand the teams. Please ask in the format 'Who will win if Team1 vs Team2?'")
