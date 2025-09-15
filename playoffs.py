import pandas as pd
import random
import numpy as np


class PlayoffSimulator:
    def __init__(self, predictor):
        """Initialize with a trained NFLGamePredictor instance"""
        self.predictor = predictor
    
    def run_monte_carlo_simulation(self, season, num_simulations=10):
        """Run multiple playoff simulations with slight variations"""
        print(f"\nRunning Monte Carlo Simulation: {num_simulations} simulations")
        print("=" * 60)
        
        # First, generate the playoff bracket that will be used for all simulations
        print("Generating projected final standings...")
        standings = self.get_projected_final_standings(season, quiet=False)
        
        if standings.empty:
            print("No standings data available!")
            return None
        
        print("\nDetermining playoff teams from projected standings...")
        playoff_teams = self.determine_playoff_teams(standings)
        
        # Check if we have enough teams
        if len(playoff_teams['AFC']) < 6 or len(playoff_teams['NFC']) < 6:
            print(f"\nWarning: Not enough playoff teams found!")
            print(f"AFC: {len(playoff_teams['AFC'])} teams, NFC: {len(playoff_teams['NFC'])} teams")
            return None
        
        # Display playoff bracket that will be used for all simulations
        print("\nPlayoff Bracket (Used for All Simulations):")
        print("=" * 50)
        for conference in ['AFC', 'NFC']:
            print(f"\n{conference} Conference:")
            for team_info in playoff_teams[conference]:
                print(f"  {team_info['seed']}. {team_info['team']} ({team_info['record']}) - {team_info['type']}")
        
        # Now run the simulations
        print(f"\n{'='*20} STARTING SIMULATIONS {'='*20}")
        
        simulation_results = []
        champion_counts = {}
        super_bowl_appearances = {}
        super_bowl_matchups = {}  
        
        for sim in range(num_simulations):
            if sim % 10 == 0 or sim < 5:  # Show progress for first 5 and every 10th simulation
                print(f"Simulation {sim + 1}/{num_simulations}...", end=" ")
            
            # Run simulation with the same playoff bracket but with noise
            results = self._simulate_playoffs_with_bracket(playoff_teams, season, add_noise=True)
            
            if results:
                simulation_results.append(results)
                
                champion = results['champion']
                champion_counts[champion] = champion_counts.get(champion, 0) + 1
                
                # Track Super Bowl appearances
                for team in results['super_bowl_teams']:
                    super_bowl_appearances[team] = super_bowl_appearances.get(team, 0) + 1
                
                # Track Super Bowl matchups
                afc_team = results['super_bowl_teams'][0]
                nfc_team = results['super_bowl_teams'][1]
                matchup = f"{afc_team} vs {nfc_team}"
                super_bowl_matchups[matchup] = super_bowl_matchups.get(matchup, 0) + 1
                
                if sim % 10 == 0 or sim < 5:
                    print(f"Champion: {champion}")
        
        # Display summary statistics
        self._display_monte_carlo_results(champion_counts, super_bowl_appearances, super_bowl_matchups, num_simulations)
        
        return simulation_results
    
    def _simulate_playoffs_with_bracket(self, playoff_teams, season, add_noise=True):
        """Simulate playoffs with a predetermined bracket"""
        try:
            # Wild Card Round
            wild_card_winners = self._simulate_wild_card_round(playoff_teams, season, add_noise)
            
            # Divisional Round
            divisional_winners = self._simulate_divisional_round(playoff_teams, wild_card_winners, season, add_noise)
            
            # Conference Championships
            super_bowl_teams = self._simulate_conference_championships(playoff_teams, divisional_winners, season, add_noise)
            
            # Super Bowl
            champion = self._simulate_super_bowl(super_bowl_teams, season, add_noise)
            
            return {
                'playoff_teams': playoff_teams,
                'wild_card_winners': wild_card_winners,
                'divisional_winners': divisional_winners,
                'super_bowl_teams': super_bowl_teams,
                'champion': champion
            }
            
        except Exception as e:
            return None
    
    def _display_monte_carlo_results(self, champion_counts, super_bowl_appearances, super_bowl_matchups, num_simulations):
        """Display Monte Carlo simulation summary"""
        print(f"\n{'='*25} MONTE CARLO RESULTS {'='*25}")
        print(f"Based on {num_simulations} simulations:\n")
        
        print("Championship Probabilities:")
        sorted_champions = sorted(champion_counts.items(), key=lambda x: x[1], reverse=True)
        for team, count in sorted_champions:
            percentage = (count / num_simulations) * 100
            print(f"  {team}: {count}/{num_simulations} ({percentage:.1f}%)")
        
        print("\nSuper Bowl Appearance Probabilities:")
        sorted_sb = sorted(super_bowl_appearances.items(), key=lambda x: x[1], reverse=True)
        for team, count in sorted_sb:
            percentage = (count / num_simulations) * 100
            print(f"  {team}: {count}/{num_simulations} ({percentage:.1f}%)")
        
        if sorted_champions:
            most_likely_champion = sorted_champions[0][0]
            win_percentage = (sorted_champions[0][1] / num_simulations) * 100
            print(f"\nMost Likely Champion: {most_likely_champion} ({win_percentage:.1f}%)")
            print("\nAll Super Bowl Matchups:")
            sorted_matchups = sorted(super_bowl_matchups.items(), key=lambda x: x[1], reverse=True)
            for matchup, count in sorted_matchups:
                percentage = (count / num_simulations) * 100
                print(f"  {matchup}: {count}/{num_simulations} ({percentage:.1f}%)")
        if sorted_matchups:
            most_likely_matchup = sorted_matchups[0][0]
            matchup_percentage = (sorted_matchups[0][1] / num_simulations) * 100
            print(f"Most Likely Super Bowl: {most_likely_matchup} ({matchup_percentage:.1f}%)")
    
    def get_projected_final_standings(self, season, quiet=False):
        """Calculate projected final standings: completed games + predicted results"""
        if not quiet:
            print(f"Calculating projected final standings for season {season}...")
        
        completed_games = self.predictor.games_df[
            (self.predictor.games_df['season'] == season) & 
            (self.predictor.games_df['week'] <= 17)
        ]
        
        if not quiet:
            print(f"Found {len(completed_games)} completed regular season games")
        
        remaining_predictions = self.predictor.predict_scheduled_games()
        
        if not quiet:
            print(f"Predicting {len(remaining_predictions)} remaining games")
        
        team_records = self._initialize_team_records()
        
        for _, game in completed_games.iterrows():
            self._update_team_record(team_records, game, is_actual=True)
        
        for _, pred in remaining_predictions.iterrows():
            predicted_game = {
                'home_team': pred['home_team'],
                'away_team': pred['away_team'],
                'home_score': 24 if pred['predicted_winner'] == pred['home_team'] else 21,
                'away_score': 21 if pred['predicted_winner'] == pred['home_team'] else 24,
                'home_win': pred['predicted_winner'] == pred['home_team'],
                'week': pred['week']
            }
            self._update_team_record(team_records, predicted_game, is_actual=False)
        
        standings_data = []
        for team, record in team_records.items():
            total_games = record['wins'] + record['losses'] + record['ties']
            win_pct = (record['wins'] + 0.5 * record['ties']) / total_games if total_games > 0 else 0
            
            standings_data.append({
                'team': team,
                'wins': record['wins'],
                'losses': record['losses'],
                'ties': record['ties'],
                'win_pct': win_pct,
                'points_for': record['points_for'],
                'points_against': record['points_against'],
                'point_diff': record['points_for'] - record['points_against'],
                'division': self.predictor.team_to_division.get(team, 'Unknown'),
                'conference': self.predictor.team_to_conference.get(team, 'Unknown'),
                'games_played': total_games
            })
        
        standings = pd.DataFrame(standings_data)
        
        if not quiet:
            print(f"\nProjected Final Standings:")
            print("Top teams by conference:")
            for conf in ['AFC', 'NFC']:
                conf_teams = standings[standings['conference'] == conf].sort_values(['win_pct', 'point_diff'], ascending=False)
                print(f"\n{conf}:")
                print(conf_teams[['team', 'wins', 'losses', 'win_pct']].head(8).to_string(index=False))
        
        return standings
    
    def _initialize_team_records(self):
        """Initialize empty records for all teams"""
        team_records = {}
        
        for division, teams in self.predictor.divisions.items():
            for team in teams:
                team_records[team] = {
                    'wins': 0,
                    'losses': 0,
                    'ties': 0,
                    'points_for': 0,
                    'points_against': 0
                }
        
        return team_records
    
    def _update_team_record(self, team_records, game, is_actual=True):
        """Update team records with a game result"""
        home_team = game['home_team']
        away_team = game['away_team']
        home_score = game['home_score']
        away_score = game['away_score']
        
        if home_team not in team_records:
            team_records[home_team] = {'wins': 0, 'losses': 0, 'ties': 0, 'points_for': 0, 'points_against': 0}
        if away_team not in team_records:
            team_records[away_team] = {'wins': 0, 'losses': 0, 'ties': 0, 'points_for': 0, 'points_against': 0}
        
        team_records[home_team]['points_for'] += home_score
        team_records[home_team]['points_against'] += away_score
        team_records[away_team]['points_for'] += away_score
        team_records[away_team]['points_against'] += home_score
        
        if home_score > away_score:
            team_records[home_team]['wins'] += 1
            team_records[away_team]['losses'] += 1
        elif away_score > home_score:
            team_records[away_team]['wins'] += 1
            team_records[home_team]['losses'] += 1
        else:
            team_records[home_team]['ties'] += 1
            team_records[away_team]['ties'] += 1
    
    def determine_playoff_teams(self, standings):
        """Determine playoff teams based on old NFL format (6 teams per conference)"""
        playoff_teams = {'AFC': [], 'NFC': []}
        
        for conference in ['AFC', 'NFC']:
            conf_standings = standings[standings['conference'] == conference]
            
            if conf_standings.empty:
                print(f"Warning: No teams found for {conference}")
                continue
            
            division_winners = []
            divisions = [f'{conference} East', f'{conference} North', 
                        f'{conference} South', f'{conference} West']
            
            for division in divisions:
                div_teams = conf_standings[conf_standings['division'] == division]
                if not div_teams.empty:
                    winner = div_teams.sort_values(['win_pct', 'point_diff'], ascending=[False, False]).iloc[0]
                    division_winners.append(winner)
            
            division_winners = sorted(division_winners, key=lambda x: (x['win_pct'], x['point_diff']), reverse=True)
            
            division_winner_teams = [w['team'] for w in division_winners]
            wild_card_candidates = conf_standings[~conf_standings['team'].isin(division_winner_teams)]
            wild_cards = wild_card_candidates.sort_values(['win_pct', 'point_diff'], ascending=[False, False]).head(2)
            
            for i, winner in enumerate(division_winners):
                playoff_teams[conference].append({
                    'team': winner['team'],
                    'seed': i + 1,
                    'type': 'Division Winner',
                    'record': f"{winner['wins']}-{winner['losses']}",
                    'win_pct': winner['win_pct']
                })
            
            for i, (_, team) in enumerate(wild_cards.iterrows()):
                playoff_teams[conference].append({
                    'team': team['team'],
                    'seed': i + 5,
                    'type': 'Wild Card',
                    'record': f"{team['wins']}-{team['losses']}",
                    'win_pct': team['win_pct']
                })
        
        return playoff_teams
    
    def simulate_playoffs(self, season, add_noise=False):
        """Simulate complete playoff bracket using projected final standings"""
        if not add_noise:
            print(f"\nSimulating Season {season} Playoffs")
            print("=" * 50)
        
        try:
            if not add_noise:
                print("Generating projected final standings...")
            standings = self.get_projected_final_standings(season, quiet=add_noise)
            
            if standings.empty:
                if not add_noise:
                    print("No standings data available!")
                return None
            
            if not add_noise:
                print("\nDetermining playoff teams from projected standings...")
            playoff_teams = self.determine_playoff_teams(standings)
            
            if len(playoff_teams['AFC']) < 6 or len(playoff_teams['NFC']) < 6:
                if not add_noise:
                    print(f"\nWarning: Not enough playoff teams found!")
                return None
            
            if not add_noise:
                print("\nPlayoff Seeding (Based on Projected Final Standings):")
                for conference in ['AFC', 'NFC']:
                    print(f"\n{conference} Conference:")
                    for team_info in playoff_teams[conference]:
                        print(f"  {team_info['seed']}. {team_info['team']} ({team_info['record']}) - {team_info['type']}")
            
            wild_card_winners = self._simulate_wild_card_round(playoff_teams, season, add_noise)
            divisional_winners = self._simulate_divisional_round(playoff_teams, wild_card_winners, season, add_noise)
            super_bowl_teams = self._simulate_conference_championships(playoff_teams, divisional_winners, season, add_noise)
            champion = self._simulate_super_bowl(super_bowl_teams, season, add_noise)
            
            return {
                'playoff_teams': playoff_teams,
                'wild_card_winners': wild_card_winners,
                'divisional_winners': divisional_winners,
                'super_bowl_teams': super_bowl_teams,
                'champion': champion
            }
            
        except Exception as e:
            if not add_noise:
                print(f"\nError in playoff simulation: {e}")
                import traceback
                traceback.print_exc()
            return None
    
    def _predict_game_with_noise(self, home_team, away_team, season, add_noise=False):
        """Predict game outcome with optional noise for Monte Carlo simulation"""
        base_prediction = self.predictor.predict_game_outcome(home_team, away_team, season, is_playoff=True)
        
        if not add_noise:
            return base_prediction
        
        base_prob = base_prediction['home_win_probability']
        
        # Use probabilistic sampling instead of just adding noise
        winner = home_team if np.random.random() < base_prob else away_team
        confidence = max(base_prob, 1 - base_prob)
        
        return {
            'home_team': home_team,
            'away_team': away_team,
            'home_win_probability': base_prob,
            'winner': winner,
            'confidence': confidence
        }
    
    def _simulate_wild_card_round(self, playoff_teams, season, add_noise=False):
        """Simulate Wild Card Round"""
        if not add_noise:
            print(f"\n{'='*20} WILD CARD ROUND {'='*20}")
        wild_card_winners = {'AFC': [], 'NFC': []}
        
        for conference in ['AFC', 'NFC']:
            teams = {team['seed']: team['team'] for team in playoff_teams[conference]}
            if not add_noise:
                print(f"\n{conference} Wild Card Games:")
            
            if not add_noise:
                print(f"Predicting: #{3} {teams[3]} vs #{6} {teams[6]}")
            game1 = self._predict_game_with_noise(teams[3], teams[6], season, add_noise)
            if not add_noise:
                print(f"Result: {game1['winner']} wins (Confidence: {game1['confidence']:.1%})")
            wild_card_winners[conference].append(game1['winner'])
            
            if not add_noise:
                print(f"Predicting: #{4} {teams[4]} vs #{5} {teams[5]}")
            game2 = self._predict_game_with_noise(teams[4], teams[5], season, add_noise)
            if not add_noise:
                print(f"Result: {game2['winner']} wins (Confidence: {game2['confidence']:.1%})")
            wild_card_winners[conference].append(game2['winner'])
        
        return wild_card_winners
    
    def _simulate_divisional_round(self, playoff_teams, wild_card_winners, season, add_noise=False):
        """Simulate Divisional Round with correct NFL bracket logic"""
        if not add_noise:
            print(f"\n{'='*20} DIVISIONAL ROUND {'='*20}")
        divisional_winners = {'AFC': [], 'NFC': []}
        
        for conference in ['AFC', 'NFC']:
            # Get the seeds for all teams
            teams_by_seed = {team['seed']: team['team'] for team in playoff_teams[conference]}
            
            # Get original seeds for wild card winners
            original_seeds = {}
            for team_info in playoff_teams[conference]:
                original_seeds[team_info['team']] = team_info['seed']
            
            wc_winners = wild_card_winners[conference]
            
            # Get seeds of wild card winners and sort them
            wc_winner_seeds = []
            for team in wc_winners:
                seed = original_seeds[team]
                wc_winner_seeds.append((seed, team))
            
            # Sort by seed (lowest seed number = higher ranking)
            wc_winner_seeds.sort(key=lambda x: x[0])
            
            if not add_noise:
                print(f"\n{conference} Divisional Games:")
                print(f"Wild card winners advancing: {[f'#{seed} {team}' for seed, team in wc_winner_seeds]}")
            
            # NFL Divisional Round Rules:
            # - #1 seed plays the LOWEST remaining seed
            # - #2 seed plays the HIGHEST remaining seed
            
            # #1 seed vs lowest remaining seed
            seed_1_team = teams_by_seed[1]
            lowest_seed, lowest_team = wc_winner_seeds[-1]  # Last in sorted list = highest seed number = lowest ranking
            
            if not add_noise:
                print(f"Game 1: #{1} {seed_1_team} vs #{lowest_seed} {lowest_team}")
            game1 = self._predict_game_with_noise(seed_1_team, lowest_team, season, add_noise)
            if not add_noise:
                print(f"Result: {game1['winner']} wins (Confidence: {game1['confidence']:.1%})")
            divisional_winners[conference].append(game1['winner'])
            
            # #2 seed vs highest remaining seed  
            seed_2_team = teams_by_seed[2]
            highest_seed, highest_team = wc_winner_seeds[0]  # First in sorted list = lowest seed number = highest ranking
            
            if not add_noise:
                print(f"Game 2: #{2} {seed_2_team} vs #{highest_seed} {highest_team}")
            game2 = self._predict_game_with_noise(seed_2_team, highest_team, season, add_noise)
            if not add_noise:
                print(f"Result: {game2['winner']} wins (Confidence: {game2['confidence']:.1%})")
            divisional_winners[conference].append(game2['winner'])
        
        return divisional_winners
    
    def _simulate_conference_championships(self, playoff_teams, divisional_winners, season, add_noise=False):
        """Simulate Conference Championships"""
        if not add_noise:
            print(f"\n{'='*20} CONFERENCE CHAMPIONSHIPS {'='*20}")
        super_bowl_teams = []
        
        for conference in ['AFC', 'NFC']:
            div_winners = divisional_winners[conference]
            
            original_seeds = {}
            for team_info in playoff_teams[conference]:
                original_seeds[team_info['team']] = team_info['seed']
            
            team1_seed = original_seeds.get(div_winners[0], 7)
            team2_seed = original_seeds.get(div_winners[1], 7)
            
            if team1_seed < team2_seed:
                home_team, away_team = div_winners[0], div_winners[1]
            else:
                home_team, away_team = div_winners[1], div_winners[0]
            
            if not add_noise:
                print(f"\n{conference} Championship:")
                print(f"Predicting: {home_team} vs {away_team}")
            game = self._predict_game_with_noise(home_team, away_team, season, add_noise)
            if not add_noise:
                print(f"Result: {game['winner']} wins (Confidence: {game['confidence']:.1%})")
            super_bowl_teams.append(game['winner'])
        
        return super_bowl_teams
    
    def _simulate_super_bowl(self, super_bowl_teams, season, add_noise=False):
        """Simulate Super Bowl"""
        if not add_noise:
            print(f"\n{'='*25} SUPER BOWL {'='*25}")
        
        if random.random() > 0.5:
            home_team, away_team = super_bowl_teams[0], super_bowl_teams[1]
        else:
            home_team, away_team = super_bowl_teams[1], super_bowl_teams[0]
        
        if not add_noise:
            print(f"\nSuper Bowl Matchup: {home_team} vs {away_team}")
        super_bowl = self._predict_game_with_noise(home_team, away_team, season, add_noise)
        if not add_noise:
            print(f"Champion: {super_bowl['winner']} (Confidence: {super_bowl['confidence']:.1%})")
        
        return super_bowl['winner']