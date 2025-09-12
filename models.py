import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class NFLGamePredictor:
    def __init__(self, raw_data):
        """
        Initialize the predictor with JSON data dictionary
        """
        self.raw_data = raw_data
        self.games_df = None
        self.team_stats = None
        self.models = {}
        self.scaler = StandardScaler()
        
        # NFL team divisions (old format)
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
        
    def process_data(self):
        """Process raw JSON data into structured DataFrames"""
        all_games = []
        
        # Extract all completed games from all seasons
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
                        'is_playoff': game.get('isPlayoff', False)
                    }
                    all_games.append(game_info)
        
        self.games_df = pd.DataFrame(all_games)
        print(f"Processed {len(self.games_df)} completed games across {len(self.raw_data['seasons'])} seasons")
        
    def calculate_team_stats(self):
        """Calculate rolling team statistics for each game"""
        team_stats = {}
        
        # Sort games by season and week
        sorted_games = self.games_df.sort_values(['season', 'week'])
        
        for idx, game in sorted_games.iterrows():
            season = game['season']
            week = game['week']
            
            # Get recent performance for both teams (last 5 games)
            home_recent = self._get_recent_performance(game['home_team'], season, week, 5)
            away_recent = self._get_recent_performance(game['away_team'], season, week, 5)
            
            # Get season performance up to this point
            home_season = self._get_season_performance(game['home_team'], season, week)
            away_season = self._get_season_performance(game['away_team'], season, week)
            
            # Head-to-head record
            h2h = self._get_head_to_head(game['home_team'], game['away_team'], season, week)
            
            game_stats = {
                'game_id': game['game_id'],
                'season': season,
                'week': week,
                'home_team': game['home_team'],
                'away_team': game['away_team'],
                'home_win': game['home_win'],
                
                # Home team stats
                'home_recent_wins': home_recent['wins'],
                'home_recent_ppg': home_recent['ppg'],
                'home_recent_papg': home_recent['papg'],
                'home_season_wins': home_season['wins'],
                'home_season_ppg': home_season['ppg'],
                'home_season_papg': home_season['papg'],
                
                # Away team stats
                'away_recent_wins': away_recent['wins'],
                'away_recent_ppg': away_recent['ppg'],
                'away_recent_papg': away_recent['papg'],
                'away_season_wins': away_season['wins'],
                'away_season_ppg': away_season['ppg'],
                'away_season_papg': away_season['papg'],
                
                # Head-to-head and other factors
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
            return {'wins': 0, 'ppg': 20, 'papg': 20}  # Default values
        
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
                points_against.append(game['away_score'])  # Fixed: was home_score
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
        
        # Only include games before current game
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
        """Train multiple prediction models"""
        X, y, feature_columns = self.prepare_features()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Initialize models
        models = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'gradient_boost': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000)
        }
        
        # Train and evaluate models
        for name, model in models.items():
            if name == 'logistic_regression':
                model.fit(X_scaled, y)
                scores = cross_val_score(model, X_scaled, y, cv=5)
            else:
                model.fit(X, y)
                scores = cross_val_score(model, X, y, cv=5)
            
            self.models[name] = model
            print(f"{name}: CV Accuracy = {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
        
        self.feature_columns = feature_columns
        
        # Feature importance for tree-based models
        if 'random_forest' in self.models:
            rf_importance = pd.DataFrame({
                'feature': feature_columns,
                'importance': self.models['random_forest'].feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\nTop 10 Most Important Features (Random Forest):")
            print(rf_importance.head(10).to_string(index=False))
    
    def find_latest_season(self):
        """Find the latest season in the data"""
        return max(int(season_id) for season_id in self.raw_data['seasons'].keys())
    
    def predict_game_outcome(self, home_team, away_team, season, week=18, is_playoff=True):
        """Predict outcome of a single game"""
        # Get team stats for prediction
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
        
        # Create feature vector in correct order
        X_game = np.array([[features[col] for col in self.feature_columns]])
        
        # Get predictions from all models
        rf_pred = self.models['random_forest'].predict_proba(X_game)[0]
        gb_pred = self.models['gradient_boost'].predict_proba(X_game)[0]
        
        X_scaled = self.scaler.transform(X_game)
        lr_pred = self.models['logistic_regression'].predict_proba(X_scaled)[0]
        
        # Ensemble prediction (average)
        home_win_prob = (rf_pred[1] + gb_pred[1] + lr_pred[1]) / 3
        
        return {
            'home_team': home_team,
            'away_team': away_team,
            'home_win_probability': home_win_prob,
            'winner': home_team if home_win_prob > 0.5 else away_team,
            'confidence': max(home_win_prob, 1 - home_win_prob)
        }
    
    def predict_latest_season_games(self):
        """Predict outcomes for remaining games in the latest season"""
        latest_season = self.find_latest_season()
        upcoming_games = []
        
        # Get scheduled games from the latest season
        for game in self.raw_data['seasons'][str(latest_season)]['games']:
            if game.get('status') == 'scheduled':
                upcoming_games.append({
                    'week': game['week'],
                    'home_team': game['homeTeamId'],
                    'away_team': game['awayTeamId'],
                    'game_id': game['id']
                })
        
        predictions = []
        
        for game in upcoming_games:
            # Calculate stats for this game
            home_recent = self._get_recent_performance(game['home_team'], latest_season, game['week'], 5)
            away_recent = self._get_recent_performance(game['away_team'], latest_season, game['week'], 5)
            home_season = self._get_season_performance(game['home_team'], latest_season, game['week'])
            away_season = self._get_season_performance(game['away_team'], latest_season, game['week'])
            h2h = self._get_head_to_head(game['home_team'], game['away_team'], latest_season, game['week'])
            
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
                'is_playoff': False,
                'ppg_differential': home_recent['ppg'] - away_recent['ppg'],
                'papg_differential': away_recent['papg'] - home_recent['papg'],
                'recent_win_diff': home_recent['wins'] - away_recent['wins'],
                'season_win_diff': home_season['wins'] - away_season['wins']
            }
            
            # Create feature vector in correct order
            X_game = np.array([[features[col] for col in self.feature_columns]])
            
            # Get predictions from all models
            rf_pred = self.models['random_forest'].predict_proba(X_game)[0]
            gb_pred = self.models['gradient_boost'].predict_proba(X_game)[0]
            
            X_scaled = self.scaler.transform(X_game)
            lr_pred = self.models['logistic_regression'].predict_proba(X_scaled)[0]
            
            # Ensemble prediction (average)
            home_win_prob = (rf_pred[1] + gb_pred[1] + lr_pred[1]) / 3
            
            predictions.append({
                'week': game['week'],
                'home_team': game['home_team'],
                'away_team': game['away_team'],
                'home_win_probability': home_win_prob,
                'predicted_winner': game['home_team'] if home_win_prob > 0.5 else game['away_team'],
                'confidence': max(home_win_prob, 1 - home_win_prob)
            })
        
        return pd.DataFrame(predictions)