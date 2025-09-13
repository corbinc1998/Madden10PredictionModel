#!/usr/bin/env python3
"""
NFL Game Predictor with Playoff Simulation
Main application entry point
"""

from adaptive_models import AdaptiveNFLPredictor, run_adaptive_predictor
from playoffs import PlayoffSimulator
from utils import (
    get_json_file_path, 
    load_nfl_data, 
    display_predictions, 
    display_simulation_summary,
    create_test_data
)


def run_regular_season_predictions():
    """Run regular season predictions only"""
    json_file_path = get_json_file_path()
    
    if not json_file_path:
        print("No data file provided. Exiting.")
        return
    
    try:
        # Load and process data
        print(f"\nLoading data from: {json_file_path}")
        raw_data = load_nfl_data(json_file_path)
        
        predictor, predictions = run_adaptive_predictor(json_file_path)
        
        print("Processing game data...")
        predictor.process_data()
        
        print("\nCalculating team statistics...")
        predictor.calculate_team_stats()
        
        print("\nTraining prediction models...")
        predictor.train_models()
        
        print("\nPredicting upcoming games...")
        predictions = predictor.predict_scheduled_games_and_playoffs()
        latest_season = max(int(s) for s in predictor.raw_data['seasons'].keys())
        
        if not display_predictions(predictions, latest_season):
            # If no upcoming games, offer to simulate playoffs
            simulate = input("\nNo regular season games remaining. Simulate playoffs instead? (y/n): ").strip().lower()
            if simulate == 'y':
                playoff_sim = PlayoffSimulator(predictor)
                results = playoff_sim.simulate_playoffs(latest_season)
                display_simulation_summary(results)
                
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


def run_playoff_simulation():
    """Run playoff simulation only"""
    json_file_path = get_json_file_path()
    
    if not json_file_path:
        print("No data file provided. Exiting.")
        return
    
    try:
        # Load and process data
        print(f"\nLoading data from: {json_file_path}")
        raw_data = load_nfl_data(json_file_path)
        
        predictor, predictions = run_adaptive_predictor(json_file_path)
        
        # print("Processing game data...")
        # predictor.process_data()
        
        # print("\nCalculating team statistics...")
        # predictor.calculate_team_stats()
        
        # print("\nTraining prediction models...")
        # predictor.train_models()
        
        # Get available seasons
        available_seasons = sorted([int(s) for s in raw_data['seasons'].keys()])
        print(f"\nAvailable seasons: {available_seasons}")
        
        # Let user choose season or simulate latest
        season_choice = input(f"\nEnter season to simulate (or press Enter for latest season {max(available_seasons)}): ").strip()
        
        if season_choice:
            try:
                season = int(season_choice)
                if season not in available_seasons:
                    print(f"Season {season} not found. Using latest season {max(available_seasons)}")
                    season = max(available_seasons)
            except ValueError:
                print("Invalid season number. Using latest season.")
                season = max(available_seasons)
        else:
            season = max(available_seasons)
        
        # Run simulation
        playoff_sim = PlayoffSimulator(predictor)
        results = playoff_sim.simulate_playoffs(season)
        
        # Display summary
        display_simulation_summary(results)
        
        # Ask if user wants to see regular season predictions too
        show_regular = input("\nShow remaining regular season game predictions? (y/n): ").strip().lower()
        if show_regular == 'y':
            predictions = predictor.predict_scheduled_games_and_playoffs()
            latest_season = max(int(s) for s in predictor.raw_data['seasons'].keys())
            display_predictions(predictions, latest_season)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


def run_both_predictions():
    """Run both regular season predictions and playoff simulation with projected standings"""
    json_file_path = get_json_file_path()
    
    if not json_file_path:
        print("No data file provided. Exiting.")
        return
    
    try:
        # Load and process data
        print(f"\nLoading data from: {json_file_path}")
        raw_data = load_nfl_data(json_file_path)
        
        predictor, predictions = run_adaptive_predictor(json_file_path)
        
        print("Processing game data...")
        predictor.process_data()
        
        print("\nCalculating team statistics...")
        predictor.calculate_team_stats()
        
        print("\nTraining prediction models...")
        predictor.train_models()
        
        latest_season = max(int(s) for s in predictor.raw_data['seasons'].keys())
        
        # Show regular season predictions first
        print("\nPredicting remaining regular season games...")
        predictions = predictor.predict_scheduled_games_and_playoffs()
        
        if not predictions.empty:
            print(f"\nRemaining Regular Season Games (Season {latest_season}):")
            print("=" * 50)
            display_predictions(predictions, latest_season)
        else:
            print("No remaining regular season games to predict.")
        
        # Then run playoff simulation with projected standings
        print("\n" + "="*60)
        print("STARTING PLAYOFF SIMULATION WITH PROJECTED STANDINGS")
        print("="*60)
        
        playoff_sim = PlayoffSimulator(predictor)
        results = playoff_sim.simulate_playoffs(latest_season)
        display_simulation_summary(results)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


def test_playoff_simulation():
    """Test the playoff simulation with minimal setup"""
    print("Testing playoff simulation with sample data...")
    test_data = create_test_data()
    
    try:
        predictor = NFLGamePredictor(test_data)
        
        predictor.process_data()
        print(f"Processed {len(predictor.games_df)} games")
        
        predictor.calculate_team_stats()
        print(f"Calculated stats for {len(predictor.team_stats)} game records")
        
        predictor.train_models()
        print("Models trained successfully")
        
        # Test playoff simulation
        playoff_sim = PlayoffSimulator(predictor)
        results = playoff_sim.simulate_playoffs(1)
        
        if results:
            print("Playoff simulation completed!")
            display_simulation_summary(results)
        else:
            print("Playoff simulation failed - check your data completeness")
        
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()


def run_monte_carlo_simulation():
    """Run Monte Carlo playoff simulation with multiple iterations"""
    json_file_path = get_json_file_path()
    
    if not json_file_path:
        print("No data file provided. Exiting.")
        return
    
    try:
        # Load and process data
        print(f"\nLoading data from: {json_file_path}")
        raw_data = load_nfl_data(json_file_path)
        
        predictor = NFLGamePredictor(raw_data)
        
        print("Processing game data...")
        predictor.process_data()
        
        print("\nCalculating team statistics...")
        predictor.calculate_team_stats()
        
        print("\nTraining prediction models...")
        predictor.train_models()
        
        # Get available seasons
        available_seasons = sorted([int(s) for s in raw_data['seasons'].keys()])
        print(f"\nAvailable seasons: {available_seasons}")
        
        # Let user choose season
        season_choice = input(f"\nEnter season to simulate (or press Enter for latest season {max(available_seasons)}): ").strip()
        
        if season_choice:
            try:
                season = int(season_choice)
                if season not in available_seasons:
                    print(f"Season {season} not found. Using latest season {max(available_seasons)}")
                    season = max(available_seasons)
            except ValueError:
                print("Invalid season number. Using latest season.")
                season = max(available_seasons)
        else:
            season = max(available_seasons)
        
        # Get number of simulations
        while True:
            try:
                num_sims = input("\nNumber of simulations to run (default 100): ").strip()
                num_sims = int(num_sims) if num_sims else 100
                if 1 <= num_sims <= 1000:
                    break
                else:
                    print("Please enter a number between 1 and 1000.")
            except ValueError:
                print("Please enter a valid number.")
        
        # Run Monte Carlo simulation
        playoff_sim = PlayoffSimulator(predictor)
        results = playoff_sim.run_monte_carlo_simulation(season, num_sims)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main function with enhanced options"""
    print("NFL Game Predictor with Playoff Simulation")
    print("==========================================")
    
    print("\nChoose an option:")
    print("1. Regular season predictions only")
    print("2. Single playoff simulation")
    print("3. Both regular season and single playoff simulation")
    print("4. Monte Carlo playoff simulation (multiple runs)")
    print("5. Test playoff simulation (debug mode)")
    
    choice = input("Enter your choice (1-5): ").strip()
    
    if choice == "1":
        run_regular_season_predictions()
    elif choice == "2":
        run_playoff_simulation()
    elif choice == "3":
        run_both_predictions()
    elif choice == "4":
        run_monte_carlo_simulation()
    elif choice == "5":
        test_playoff_simulation()
    else:
        print("Invalid choice. Please run again and select 1-5.")


if __name__ == "__main__":
    main()