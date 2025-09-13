import pandas as pd
import numpy as np
import json

class NumpyEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, (np.bool_,)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

import pickle
import os
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
import warnings
warnings.filterwarnings('ignore')


class AdaptiveNFLPredictor:
    def __init__(self, raw_data, config_path="predictor_config.json", predictions_dir="predictions"):
        """
        Initialize the adaptive predictor
        """
        self.raw_data = raw_data
        self.config_path = config_path
        self.predictions_dir = predictions_dir
        self.games_df = None
        self.team_stats = None
        self.models = {}
        self.scaler = StandardScaler()
        self.stored_predictions = {}
        self.performance_history = []
        self.config = self.load_config()
        self.update_counter = 0

        self.divisions = {
            'AFC East': ['ne', 'buf', 'mia', 'nyj'],
            'AFC North': ['pit', 'bal', 'cin', 'cle'],
            'AFC South': ['ind', 'jax', 'hou', 'ten'],
            'AFC West': ['den', 'kc', 'oak', 'sd'],
            'NFC East': ['dal', 'nyg', 'phi', 'was'],
            'NFC North': ['gb', 'chi', 'det', 'min'],
            'NFC South': ['no', 'atl', 'car', 'tb'],
            'NFC West': ['sf', 'sea', 'ari', 'stl']
        }

        # Reverse mapping for team to division/conference
        self.team_to_division = {}
        self.team_to_conference = {}
        for division, teams in self.divisions.items():
            conference = division.split()[0]  # AFC or NFC
            for team in teams:
                self.team_to_division[team] = division
                self.team_to_conference[team] = conference
        
        # Create predictions directory if it doesn't exist (MOVED OUTSIDE THE LOOP)
        os.makedirs(predictions_dir, exist_ok=True)
        
        # Load most recent predictions if they exist (MOVED OUTSIDE THE LOOP)
        self.load_most_recent_predictions()

    def predict_game_outcome(self, home_team, away_team, season, week=18, is_playoff=True):
            """Backward compatibility wrapper for playoff simulation"""
            result = self.predict_game_with_features(home_team, away_team, season, week, is_playoff)
            
            # Convert to the format expected by PlayoffSimulator
            return {
                'home_team': home_team,
                'away_team': away_team,
                'home_win_probability': result['home_win_probability'],
                'winner': result['predicted_winner'],  # Convert key name
                'confidence': result['confidence']
            }

    def find_latest_season(self):
        """Backward compatibility wrapper"""
        return max(int(season_id) for season_id in self.raw_data['seasons'].keys())
        
    def load_config(self):
        """Load configuration or create default"""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            # Default configuration
            return {
                "model_weights": {
                    "random_forest": 0.4,
                    "gradient_boost": 0.35,
                    "logistic_regression": 0.25
                },
                "feature_weights": {},
                "learning_rate": 0.1,
                "min_games_for_update": 5,
                "performance_window": 20,  # Number of recent predictions to evaluate
                "last_update": None,
                "total_predictions": 0,
                "correct_predictions": 0,
                "update_counter": 0,
                "prediction_files": []  # Track all prediction files
            }
    
    def save_config(self):
        """Save current configuration"""
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=2, cls=NumpyEncoder)


    
    def load_most_recent_predictions(self):
        """Load the most recent predictions from the predictions directory"""
        import glob
        
        # Get all prediction files
        prediction_files = glob.glob(os.path.join(self.predictions_dir, "*.json"))
        
        if not prediction_files:
            self.stored_predictions = {}
            return
        
        # Find the most recent prediction file
        most_recent_file = max(prediction_files, key=os.path.getctime)
        
        try:
            with open(most_recent_file, 'r') as f:
                self.stored_predictions = json.load(f)
            print(f"Loaded predictions from: {os.path.basename(most_recent_file)}")
        except (FileNotFoundError, json.JSONDecodeError):
            self.stored_predictions = {}
    
    def get_prediction_filename(self, update_number=None):
        """Generate filename for predictions based on update number"""
        if update_number is None:
            update_number = self.config.get('update_counter', 0)
        
        if update_number == 0:
            return os.path.join(self.predictions_dir, "initial_predictions.json")
        else:
            return os.path.join(self.predictions_dir, f"after_week_{update_number}_predictions.json")

    
    def save_predictions_with_timestamp(self, predictions_data=None):
        """Save predictions to timestamped file"""
        if predictions_data is None:
            predictions_data = self.stored_predictions
        
        filename = self.get_prediction_filename()
        
        # Save the predictions
        with open(filename, 'w') as f:
            json.dump(predictions_data, f, indent=2, cls=NumpyEncoder)

        
        # Update config with filename
        if filename not in self.config.get('prediction_files', []):
            if 'prediction_files' not in self.config:
                self.config['prediction_files'] = []
            self.config['prediction_files'].append(filename)
        
        print(f"Saved predictions to: {os.path.basename(filename)}")
        return filename
    
    def process_data(self):
        """Process raw JSON data into structured DataFrames"""
        all_games = []
        
        for season_id, season_data in self.raw_data['seasons'].items():
            for game in season_data['games']:
                if game.get('completed', False) and game.get('homeScore') is not None:
                    game_info = {
                        'season': int(season_id),
                        'week': game['week'],
                        'home_team': game['homeTeamId'],
                        'away_team': game['awayTeamId'],
                        'home_score': game['homeScore'],
                        'away_score': game['awayScore'],
                        'home_win': 1 if game['homeScore'] > game['awayScore'] else 0,
                        'game_id': game['id'],
                        'is_playoff': game.get('isPlayoff', False),
                        'date': game.get('date', ''),
                        'completed': game.get('completed', False)
                    }
                    all_games.append(game_info)
        
        self.games_df = pd.DataFrame(all_games)
        print(f"Processed {len(self.games_df)} completed games")
    
    def calculate_team_stats(self):
        """Calculate rolling team statistics for each game"""
        team_stats = {}
        sorted_games = self.games_df.sort_values(['season', 'week'])
        
        for idx, game in sorted_games.iterrows():
            season = game['season']
            week = game['week']
            
            # Get performance metrics
            home_recent = self._get_recent_performance(game['home_team'], season, week, 5)
            away_recent = self._get_recent_performance(game['away_team'], season, week, 5)
            home_season = self._get_season_performance(game['home_team'], season, week)
            away_season = self._get_season_performance(game['away_team'], season, week)
            h2h = self._get_head_to_head(game['home_team'], game['away_team'], season, week)
            
            game_stats = {
                'game_id': game['game_id'],
                'season': season,
                'week': week,
                'home_team': game['home_team'],
                'away_team': game['away_team'],
                'home_win': game['home_win'],
                'home_recent_wins': home_recent['wins'],
                'home_recent_ppg': home_recent['ppg'],
                'home_recent_papg': home_recent['papg'],
                'home_season_wins': home_season['wins'],
                'home_season_ppg': home_season['ppg'],
                'home_season_papg': home_season['papg'],
                'away_recent_wins': away_recent['wins'],
                'away_recent_ppg': away_recent['ppg'],
                'away_recent_papg': away_recent['papg'],
                'away_season_wins': away_season['wins'],
                'away_season_ppg': away_season['ppg'],
                'away_season_papg': away_season['papg'],
                'h2h_home_wins': h2h['home_wins'],
                'h2h_total_games': h2h['total_games'],
                'is_playoff': game['is_playoff']
            }
            
            team_stats[game['game_id']] = game_stats
        
        self.team_stats = pd.DataFrame(list(team_stats.values()))
    
    def _get_recent_performance(self, team, season, week, num_games):
        """Get team's performance in last N games before current game"""
        team_games = self.games_df[
            ((self.games_df['home_team'] == team) | (self.games_df['away_team'] == team)) &
            ((self.games_df['season'] < season) | 
             ((self.games_df['season'] == season) & (self.games_df['week'] < week)))
        ].tail(num_games)
        
        if len(team_games) == 0:
            return {'wins': 0, 'ppg': 20, 'papg': 20}
        
        wins = 0
        points_for = []
        points_against = []
        
        for _, game in team_games.iterrows():
            if game['home_team'] == team:
                points_for.append(game['home_score'])
                points_against.append(game['away_score'])
                if game['home_win']:
                    wins += 1
            else:
                points_for.append(game['away_score'])
                points_against.append(game['home_score'])
                if not game['home_win']:
                    wins += 1
        
        return {
            'wins': wins,
            'ppg': np.mean(points_for),
            'papg': np.mean(points_against)
        }
    
    def _get_season_performance(self, team, season, week):
        """Get team's season performance up to current week"""
        team_games = self.games_df[
            ((self.games_df['home_team'] == team) | (self.games_df['away_team'] == team)) &
            (self.games_df['season'] == season) & (self.games_df['week'] < week)
        ]
        
        if len(team_games) == 0:
            return {'wins': 0, 'ppg': 20, 'papg': 20}
        
        wins = 0
        points_for = []
        points_against = []
        
        for _, game in team_games.iterrows():
            if game['home_team'] == team:
                points_for.append(game['home_score'])
                points_against.append(game['away_score'])
                if game['home_win']:
                    wins += 1
            else:
                points_for.append(game['away_score'])
                points_against.append(game['home_score'])
                if not game['home_win']:
                    wins += 1
        
        return {
            'wins': wins,
            'ppg': np.mean(points_for),
            'papg': np.mean(points_against)
        }
    
    def _get_head_to_head(self, home_team, away_team, season, week):
        """Get historical head-to-head record"""
        h2h_games = self.games_df[
            ((self.games_df['home_team'] == home_team) & (self.games_df['away_team'] == away_team)) |
            ((self.games_df['home_team'] == away_team) & (self.games_df['away_team'] == home_team))
        ]
        
        h2h_games = h2h_games[
            (h2h_games['season'] < season) | 
            ((h2h_games['season'] == season) & (h2h_games['week'] < week))
        ]
        
        if len(h2h_games) == 0:
            return {'home_wins': 0, 'total_games': 0}
        
        home_wins = 0
        for _, game in h2h_games.iterrows():
            if ((game['home_team'] == home_team and game['home_win']) or
                (game['away_team'] == home_team and not game['home_win'])):
                home_wins += 1
        
        return {
            'home_wins': home_wins,
            'total_games': len(h2h_games)
        }
    
    def prepare_features(self):
        """Prepare feature matrix for machine learning"""
        feature_columns = [
            'home_recent_wins', 'home_recent_ppg', 'home_recent_papg',
            'home_season_wins', 'home_season_ppg', 'home_season_papg',
            'away_recent_wins', 'away_recent_ppg', 'away_recent_papg',
            'away_season_wins', 'away_season_ppg', 'away_season_papg',
            'h2h_home_wins', 'h2h_total_games', 'is_playoff'
        ]
        
        # Create derived features
        self.team_stats['ppg_differential'] = (self.team_stats['home_recent_ppg'] - 
                                              self.team_stats['away_recent_ppg'])
        self.team_stats['papg_differential'] = (self.team_stats['away_recent_papg'] - 
                                               self.team_stats['home_recent_papg'])
        self.team_stats['recent_win_diff'] = (self.team_stats['home_recent_wins'] - 
                                             self.team_stats['away_recent_wins'])
        self.team_stats['season_win_diff'] = (self.team_stats['home_season_wins'] - 
                                             self.team_stats['away_season_wins'])
        
        feature_columns.extend(['ppg_differential', 'papg_differential', 
                               'recent_win_diff', 'season_win_diff'])
        
        X = self.team_stats[feature_columns].fillna(0)
        y = self.team_stats['home_win']
        
        return X, y, feature_columns
    
    def train_models(self):
        """Train models with current configuration"""
        X, y, feature_columns = self.prepare_features()
        X_scaled = self.scaler.fit_transform(X)
        
        # Initialize models
        models = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'gradient_boost': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000)
        }
        
        # Train models
        for name, model in models.items():
            if name == 'logistic_regression':
                model.fit(X_scaled, y)
                scores = cross_val_score(model, X_scaled, y, cv=5)
            else:
                model.fit(X, y)
                scores = cross_val_score(model, X, y, cv=5)
            
            self.models[name] = model
            print(f"{name}: CV Accuracy = {scores.mean():.3f}")
        
        self.feature_columns = feature_columns
    
    def predict_game_with_features(self, home_team, away_team, season, week=18, is_playoff=False):
        """Predict a single game and return prediction with features"""
        # Get team stats
        home_recent = self._get_recent_performance(home_team, season, week, 5)
        away_recent = self._get_recent_performance(away_team, season, week, 5)
        home_season = self._get_season_performance(home_team, season, week)
        away_season = self._get_season_performance(away_team, season, week)
        h2h = self._get_head_to_head(home_team, away_team, season, week)
        
        # Create feature vector
        features = {
            'home_recent_wins': home_recent['wins'],
            'home_recent_ppg': home_recent['ppg'],
            'home_recent_papg': home_recent['papg'],
            'home_season_wins': home_season['wins'],
            'home_season_ppg': home_season['ppg'],
            'home_season_papg': home_season['papg'],
            'away_recent_wins': away_recent['wins'],
            'away_recent_ppg': away_recent['ppg'],
            'away_recent_papg': away_recent['papg'],
            'away_season_wins': away_season['wins'],
            'away_season_ppg': away_season['ppg'],
            'away_season_papg': away_season['papg'],
            'h2h_home_wins': h2h['home_wins'],
            'h2h_total_games': h2h['total_games'],
            'is_playoff': is_playoff,
            'ppg_differential': home_recent['ppg'] - away_recent['ppg'],
            'papg_differential': away_recent['papg'] - home_recent['papg'],
            'recent_win_diff': home_recent['wins'] - away_recent['wins'],
            'season_win_diff': home_season['wins'] - away_season['wins']
        }
        
        # Create feature vector
        X_game = np.array([[features[col] for col in self.feature_columns]])
        
        # Get individual model predictions
        rf_pred = self.models['random_forest'].predict_proba(X_game)[0]
        gb_pred = self.models['gradient_boost'].predict_proba(X_game)[0]
        
        X_scaled = self.scaler.transform(X_game)
        lr_pred = self.models['logistic_regression'].predict_proba(X_scaled)[0]
        
        # Apply adaptive weights
        weights = self.config['model_weights']
        home_win_prob = (
            weights['random_forest'] * rf_pred[1] +
            weights['gradient_boost'] * gb_pred[1] +
            weights['logistic_regression'] * lr_pred[1]
        )
        
        return {
            'home_win_probability': home_win_prob,
            'predicted_winner': home_team if home_win_prob > 0.5 else away_team,
            'confidence': max(home_win_prob, 1 - home_win_prob),
            'features_used': features,
            'individual_predictions': {
                'random_forest': rf_pred[1],
                'gradient_boost': gb_pred[1],
                'logistic_regression': lr_pred[1]
            }
        }
    
    def predict_scheduled_games(self):
        """Predict all scheduled games and store results with comparison to previous predictions"""
        latest_season = max(int(s) for s in self.raw_data['seasons'].keys())
        scheduled_games = []
        
        # Load previous predictions for comparison if they exist
        previous_predictions = {}
        if self.config.get('prediction_files'):
            last_file = self.config['prediction_files'][-1] if self.config['prediction_files'] else None
            if last_file and os.path.exists(last_file):
                try:
                    with open(last_file, 'r') as f:
                        previous_predictions = json.load(f)
                except:
                    pass
        
        # Find scheduled games
        for game in self.raw_data['seasons'][str(latest_season)]['games']:
            if game.get('status') == 'scheduled':
                prediction = self.predict_game_with_features(
                    game['homeTeamId'], 
                    game['awayTeamId'], 
                    latest_season, 
                    game['week']
                )
                
                # Store prediction with comparison to previous
                game_id = game['id']
                
                prediction_record = {
                    'prediction': prediction,
                    'game_info': {
                        'home_team': game['homeTeamId'],
                        'away_team': game['awayTeamId'],
                        'week': game['week'],
                        'season': latest_season,
                        'date': game.get('date', '')
                    },
                    'prediction_date': datetime.now().isoformat(),
                    'update_number': self.config.get('update_counter', 0),
                    'evaluated': False
                }
                
                # Add comparison to previous prediction if it exists
                if game_id in previous_predictions:
                    prev_pred = previous_predictions[game_id]['prediction']
                    prediction_record['previous_prediction'] = {
                        'winner': prev_pred['predicted_winner'],
                        'confidence': prev_pred['confidence'],
                        'home_win_prob': prev_pred['home_win_probability']
                    }
                    
                    # Flag if prediction changed
                    prediction_record['prediction_changed'] = (
                        prev_pred['predicted_winner'] != prediction['predicted_winner']
                    )
                
                self.stored_predictions[game_id] = prediction_record
                
                scheduled_games.append({
                    'game_id': game_id,
                    'week': game['week'],
                    'home_team': game['homeTeamId'],
                    'away_team': game['awayTeamId'],
                    'predicted_winner': prediction['predicted_winner'],
                    'confidence': prediction['confidence'],
                    'home_win_prob': prediction['home_win_probability'],
                    'prediction_changed': prediction_record.get('prediction_changed', False),
                    'update_number': self.config.get('update_counter', 0)
                })
        
        # Save predictions to timestamped file
        self.save_predictions_with_timestamp()
        
        return pd.DataFrame(scheduled_games)
    
    def evaluate_predictions(self):
        """Evaluate predictions against actual results and update model weights"""
        correct = 0
        total = 0
        recent_performance = []
        
        # Check stored predictions against actual results
        for game_id, pred_data in self.stored_predictions.items():
            if pred_data.get('evaluated', True):  # default True means "skip" if not a normal game prediction
                     continue

                
            # Look for actual result in games_df
            actual_game = self.games_df[self.games_df['game_id'] == game_id]
            
            if len(actual_game) > 0:
                actual_result = actual_game.iloc[0]
                predicted_home_win = pred_data['prediction']['home_win_probability'] > 0.5
                actual_home_win = actual_result['home_win'] == 1
                
                is_correct = predicted_home_win == actual_home_win
                
                # Calculate prediction quality metrics
                prob_score = pred_data['prediction']['home_win_probability']
                if not actual_home_win:
                    prob_score = 1 - prob_score
                
                performance_record = {
                    'game_id': game_id,
                    'correct': is_correct,
                    'confidence': pred_data['prediction']['confidence'],
                    'probability_score': prob_score,
                    'week': pred_data['game_info']['week'],
                    'individual_predictions': pred_data['prediction']['individual_predictions'],
                    'actual_home_win': actual_home_win,
                    'update_number': pred_data.get('update_number', 0)
                }
                
                recent_performance.append(performance_record)
                self.stored_predictions[game_id]['evaluated'] = True
                self.stored_predictions[game_id]['actual_result'] = {
                    'home_win': actual_home_win,
                    'correct_prediction': is_correct,
                    'evaluation_date': datetime.now().isoformat()
                }
                
                if is_correct:
                    correct += 1
                total += 1
        
        if total > 0:
            accuracy = correct / total
            print(f"Weekly evaluation: {correct}/{total} correct ({accuracy:.1%})")
            
            # Update model weights based on individual model performance
            self._update_model_weights(recent_performance)
            
            # Update config
            self.config['total_predictions'] += total
            self.config['correct_predictions'] += correct
            self.config['last_update'] = datetime.now().isoformat()
            
            self.performance_history.extend(recent_performance)
            self.save_config()
            
            # Save updated predictions with evaluation results
            self.save_predictions_with_timestamp()
            
            return accuracy
        
        return None

    def predict_scheduled_games_and_playoffs(self):
        """Extended version that includes playoff predictions"""
        # Get regular season predictions
        regular_predictions = self.predict_scheduled_games()
        
        # Add playoff simulation
        try:
            from playoffs import PlayoffSimulator
            playoff_sim = PlayoffSimulator(self)
            latest_season = self.find_latest_season()

            # ✅ Run playoff simulation ONCE
            playoff_results = playoff_sim.simulate_playoffs(latest_season)

            if playoff_results:
                playoff_prediction = {
                    'prediction_type': 'playoff_simulation',
                    'simulation_results': playoff_results,
                    'simulation_date': datetime.now().isoformat(),
                    'season': latest_season
                }

                # Save into stored predictions
                self.stored_predictions[f'playoff_simulation_{latest_season}'] = playoff_prediction
                self.save_predictions_with_timestamp()

                # ✅ Immediately show the SAME results you saved
                from utils import display_simulation_summary
                display_simulation_summary(playoff_results)

        except Exception as e:
            print(f"Playoff simulation failed: {e}")

        return regular_predictions
    
    def _update_model_weights(self, recent_performance):
        """Update model weights based on recent performance"""
        if len(recent_performance) < self.config['min_games_for_update']:
            return
        
        # Calculate individual model accuracies
        model_scores = {'random_forest': [], 'gradient_boost': [], 'logistic_regression': []}
        
        for perf in recent_performance:
            actual_prob = 1.0 if perf['actual_home_win'] else 0.0
            
            for model_name in model_scores.keys():
                pred_prob = perf['individual_predictions'][model_name]
                # Use Brier score (lower is better)
                brier_score = (pred_prob - actual_prob) ** 2
                model_scores[model_name].append(1 - brier_score)  # Convert to "accuracy-like" metric
        
        # Calculate average scores
        avg_scores = {name: np.mean(scores) for name, scores in model_scores.items()}
        
        # Update weights with exponential smoothing
        learning_rate = self.config['learning_rate']
        total_score = sum(avg_scores.values())
        
        if total_score > 0:
            new_weights = {name: score / total_score for name, score in avg_scores.items()}
            
            # Apply exponential smoothing to weights
            for model_name in self.config['model_weights']:
                old_weight = self.config['model_weights'][model_name]
                new_weight = new_weights[model_name]
                self.config['model_weights'][model_name] = float(
            (1 - learning_rate) * old_weight + learning_rate * new_weight
        )

        
        print("Updated model weights:", self.config['model_weights'])
    
    def weekly_update_cycle(self):
        """Perform weekly update cycle"""
        print("Starting weekly update cycle...")
        
        # 1. Increment update counter
        self.config['update_counter'] = self.config.get('update_counter', 0) + 1
        print(f"Update #{self.config['update_counter']}")
        
        # 2. Reprocess data to include newly completed games
        print("Reprocessing data...")
        self.process_data()
        self.calculate_team_stats()
        
        # 3. Evaluate previous predictions
        print("Evaluating previous predictions...")
        accuracy = self.evaluate_predictions()
        if accuracy is not None:
            print(f"Recent prediction accuracy: {accuracy:.1%}")
        
        # 4. Retrain models with updated data
        print("Retraining models...")
        self.train_models()
        
        # 5. Generate new predictions for remaining scheduled games
        print("Generating updated predictions...")
        new_predictions = self.predict_scheduled_games()
        
        # 6. Show prediction changes summary
        self._show_prediction_changes_summary(new_predictions)
        
        print(f"Updated predictions for {len(new_predictions)} games")
        
        return new_predictions
    
    def _show_prediction_changes_summary(self, new_predictions):
        """Show summary of how predictions changed"""
        if not new_predictions.empty:
            changed_predictions = new_predictions[new_predictions['prediction_changed'] == True]
            
            if len(changed_predictions) > 0:
                print(f"\nPrediction Changes Summary:")
                print(f"Changed predictions: {len(changed_predictions)}/{len(new_predictions)}")
                
                for _, pred in changed_predictions.iterrows():
                    print(f"  Week {pred['week']}: {pred['home_team']} vs {pred['away_team']} -> Now predicting {pred['predicted_winner']}")
            else:
                print("No prediction changes from previous update.")
    
    def compare_prediction_files(self, file1=None, file2=None):
        """Compare two prediction files to see what changed"""
        if file1 is None or file2 is None:
            # Get last two prediction files
            if len(self.config.get('prediction_files', [])) < 2:
                print("Need at least 2 prediction files to compare")
                return
            
            files = sorted(self.config['prediction_files'])
            file1, file2 = files[-2], files[-1]
        
        try:
            with open(file1, 'r') as f:
                pred1 = json.load(f)
            with open(file2, 'r') as f:
                pred2 = json.load(f)
        except FileNotFoundError as e:
            print(f"Could not load file: {e}")
            return
        
        print(f"\nComparing predictions:")
        print(f"  {os.path.basename(file1)}")
        print(f"  {os.path.basename(file2)}")
        print("-" * 50)
        
        changes = 0
        for game_id in pred1.keys():
            if game_id in pred2:
                old_winner = pred1[game_id]['prediction']['predicted_winner']
                new_winner = pred2[game_id]['prediction']['predicted_winner']
                
                if old_winner != new_winner:
                    game_info = pred2[game_id]['game_info']
                    print(f"Week {game_info['week']}: {game_info['home_team']} vs {game_info['away_team']}")
                    print(f"  Changed from {old_winner} to {new_winner}")
                    changes += 1
        
        print(f"\nTotal prediction changes: {changes}")
        
        return changes
    
    def get_performance_summary(self):
        """Get overall performance summary with prediction file tracking"""
        total = self.config['total_predictions']
        correct = self.config['correct_predictions']
        
        print(f"\nOverall Performance Summary:")
        print(f"Total predictions evaluated: {total}")
        if total > 0:
            overall_accuracy = correct / total
            print(f"Correct predictions: {correct}")
            print(f"Overall accuracy: {overall_accuracy:.1%}")
        else:
            print("No predictions have been evaluated yet.")
        
        print(f"Current model weights: {self.config['model_weights']}")
        print(f"Update counter: {self.config.get('update_counter', 0)}")
        print(f"Last update: {self.config.get('last_update', 'Never')}")
        
        # Show prediction files created
        if self.config.get('prediction_files'):
            print(f"\nPrediction files created:")
            for i, file in enumerate(self.config['prediction_files']):
                file_basename = os.path.basename(file)
                print(f"  {i+1}. {file_basename}")
        else:
            print("No prediction files created yet.")
    
    def get_all_predictions_history(self):
        """Load and return all prediction history from all files"""
        all_predictions = {}
        
        for file_path in self.config.get('prediction_files', []):
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r') as f:
                        predictions = json.load(f)
                    
                    file_basename = os.path.basename(file_path)
                    all_predictions[file_basename] = predictions
                except:
                    continue
        
        return all_predictions


# Usage example
def run_adaptive_predictor(json_file_path):
    """Run the adaptive predictor system"""
    # Load data
    with open(json_file_path, 'r') as f:
        raw_data = json.load(f)
    
    # Initialize predictor
    predictor = AdaptiveNFLPredictor(raw_data)
    
    print("Processing initial data...")
    predictor.process_data()
    predictor.calculate_team_stats()
    
    print("Training initial models...")
    predictor.train_models()
    
   # Check if this is a weekly update or initial run
    if predictor.config['update_counter'] == 0:
        # Count completed games
        completed_games = predictor.games_df[predictor.games_df['completed'] == True]

        if completed_games.empty:
            print("No completed games yet. Models trained, but predictions will start after Week 1.")
            predictions = None
        else:
            print("First completed week detected — generating predictions...")
            predictions = predictor.weekly_update_cycle()
    else:
        print("Performing weekly update...")
        predictions = predictor.weekly_update_cycle()

    
    # Show performance summary
    predictor.get_performance_summary()
    
    # Show recent prediction changes if this was an update
    if predictor.config['update_counter'] > 0:
        print("\nTo compare prediction changes, run:")
        print("predictor.compare_prediction_files()")
    
    return predictor, predictions


def analyze_prediction_evolution(predictor):
    """Analyze how predictions evolved over time"""
    all_predictions = predictor.get_all_predictions_history()
    
    if len(all_predictions) < 2:
        print("Need at least 2 prediction files to analyze evolution")
        return
    
    print("Prediction Evolution Analysis:")
    print("=" * 50)
    
    # Track how often predictions changed for each game
    game_changes = {}
    file_names = sorted(all_predictions.keys())
    
    for i in range(1, len(file_names)):
        prev_file = file_names[i-1]
        curr_file = file_names[i]
        
        changes = 0
        total_games = 0
        
        for game_id in all_predictions[curr_file]:
            if game_id in all_predictions[prev_file]:
                prev_winner = all_predictions[prev_file][game_id]['prediction']['predicted_winner']
                curr_winner = all_predictions[curr_file][game_id]['prediction']['predicted_winner']
                
                if game_id not in game_changes:
                    game_changes[game_id] = 0
                
                if prev_winner != curr_winner:
                    game_changes[game_id] += 1
                    changes += 1
                
                total_games += 1
        
        change_rate = changes / total_games if total_games > 0 else 0
        print(f"{prev_file} -> {curr_file}: {changes}/{total_games} changed ({change_rate:.1%})")
    
    # Show most volatile games
    if game_changes:
        print(f"\nMost frequently changed predictions:")
        sorted_changes = sorted(game_changes.items(), key=lambda x: x[1], reverse=True)
        
        for game_id, change_count in sorted_changes[:5]:
            if change_count > 0:
                # Get game info from most recent file
                recent_file = file_names[-1]
                game_info = all_predictions[recent_file][game_id]['game_info']
                print(f"  Week {game_info['week']}: {game_info['home_team']} vs {game_info['away_team']} - Changed {change_count} times")


# Additional utility functions
def load_and_display_predictions(prediction_file):
    """Load and display a specific prediction file"""
    try:
        with open(prediction_file, 'r') as f:
            predictions = json.load(f)
        
        print(f"\nPredictions from {os.path.basename(prediction_file)}:")
        print("=" * 60)
        
        # Sort by week
        games = list(predictions.values())
        games.sort(key=lambda x: x['game_info']['week'])
        
        for game in games:
            info = game['game_info']
            pred = game['prediction']
            
            print(f"Week {info['week']}: {info['home_team']} vs {info['away_team']}")
            print(f"  Predicted Winner: {pred['predicted_winner']}")
            print(f"  Confidence: {pred['confidence']:.1%}")
            
            if 'actual_result' in game:
                result = "✓" if game['actual_result']['correct_prediction'] else "✗"
                print(f"  Actual Result: {result}")
            
            if 'prediction_changed' in game and game['prediction_changed']:
                print(f"  🔄 Prediction changed from previous update")
            
            print()
    
    except FileNotFoundError:
        print(f"File not found: {prediction_file}")
    except json.JSONDecodeError:
        print(f"Invalid JSON file: {prediction_file}")


if __name__ == "__main__":
    # Example usage
    json_file = "your_nfl_data.json"  # Replace with your file
    
    if os.path.exists(json_file):
        predictor, predictions = run_adaptive_predictor(json_file)
        
        # Optional: Run analysis if you have multiple prediction files
        if len(predictor.config.get('prediction_files', [])) > 1:
            analyze_prediction_evolution(predictor)
    else:
        print(f"Please ensure your data file '{json_file}' exists.")