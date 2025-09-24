import os
import sys
from datetime import datetime
from contextlib import redirect_stdout
import io

class TextDocumentGenerator:
    def __init__(self, output_dir="output_documents"):
        """Initialize the text document generator"""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def generate_filename(self, base_name, extension=".txt"):
        """Generate timestamped filename"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(self.output_dir, f"{base_name}_{timestamp}{extension}")
    
    def capture_and_display_output(self, func, filename, *args, **kwargs):
        """Capture function output to file while also displaying to console"""
        # Create string buffer to capture output
        output_buffer = io.StringIO()
        
        # Create a custom writer that writes to both console and buffer
        class TeeWriter:
            def __init__(self, buffer, console):
                self.buffer = buffer
                self.console = console
                
            def write(self, text):
                self.buffer.write(text)
                self.console.write(text)
                
            def flush(self):
                self.buffer.flush()
                self.console.flush()
        
        # Save original stdout
        original_stdout = sys.stdout
        
        # Create tee writer that writes to both buffer and console
        tee_writer = TeeWriter(output_buffer, original_stdout)
        
        try:
            # Redirect stdout to tee writer
            sys.stdout = tee_writer
            result = func(*args, **kwargs)
        except Exception as e:
            print(f"Error occurred: {str(e)}")
            import traceback
            traceback.print_exc()
            result = None
        finally:
            # Restore original stdout
            sys.stdout = original_stdout
        
        # Get the captured output
        captured_output = output_buffer.getvalue()
        
        # Write to file
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"NFL Game Predictor Report\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
            f.write(captured_output)
        
        print(f"\nReport also saved to: {filename}")
        return result, filename
    
    def run_regular_season_to_file(self, json_file_path):
        """Run regular season predictions and save to file"""
        filename = self.generate_filename("regular_season_predictions")
        
        def run_predictions():
            from adaptive_models import run_adaptive_predictor
            from utils import display_predictions, load_nfl_data
            
            print(f"Loading data from: {json_file_path}")
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
                print("\nNo regular season games remaining.")
                
            return predictor, predictions
        
        return self.capture_and_display_output(run_predictions, filename)
    
    def run_playoff_simulation_to_file(self, json_file_path, season=None):
        """Run playoff simulation and save to file"""
        filename = self.generate_filename("playoff_simulation")
        
        def run_simulation():
            from adaptive_models import run_adaptive_predictor
            from playoffs import PlayoffSimulator
            from utils import display_simulation_summary, load_nfl_data
            
            print(f"Loading data from: {json_file_path}")
            raw_data = load_nfl_data(json_file_path)
            
            predictor, predictions = run_adaptive_predictor(json_file_path)
            
            # Get available seasons
            available_seasons = sorted([int(s) for s in raw_data['seasons'].keys()])
            print(f"Available seasons: {available_seasons}")
            
            if season is None:
                season = max(available_seasons)
            
            print(f"Simulating playoffs for season {season}")
            
            # Run simulation
            playoff_sim = PlayoffSimulator(predictor)
            results = playoff_sim.simulate_playoffs(season)
            
            # Display summary
            if results:
                display_simulation_summary(results)
            else:
                print("Playoff simulation failed - check your data completeness")
                
            return results
        
        return self.capture_and_display_output(run_simulation, filename)
    
    def run_both_predictions_to_file(self, json_file_path):
        """Run both regular season and playoffs, save to file"""
        filename = self.generate_filename("complete_season_analysis")
        
        def run_both():
            from adaptive_models import run_adaptive_predictor
            from playoffs import PlayoffSimulator
            from utils import display_predictions, display_simulation_summary, load_nfl_data
            
            print(f"Loading data from: {json_file_path}")
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
            if results:
                display_simulation_summary(results)
            
            return predictor, predictions, results
        
        return self.capture_and_display_output(run_both, filename)
    
    def run_monte_carlo_to_file(self, json_file_path, season=None, num_simulations=100):
        """Run Monte Carlo simulation and save to file"""
        filename = self.generate_filename("monte_carlo_simulation")
        
        def run_monte_carlo():
            from adaptive_models import run_adaptive_predictor
            from playoffs import PlayoffSimulator
            from utils import load_nfl_data
            
            print(f"Loading data from: {json_file_path}")
            raw_data = load_nfl_data(json_file_path)
            
            predictor, _ = run_adaptive_predictor(json_file_path)
            
            # Get available seasons
            available_seasons = sorted([int(s) for s in raw_data['seasons'].keys()])
            print(f"Available seasons: {available_seasons}")
            
            if season is None:
                season = max(available_seasons)
            
            print(f"Running Monte Carlo simulation for season {season}")
            print(f"Number of simulations: {num_simulations}")
            
            # Run Monte Carlo simulation
            playoff_sim = PlayoffSimulator(predictor)
            results = playoff_sim.run_monte_carlo_simulation(season, num_simulations)
            
            return results
        
        return self.capture_and_display_output(run_monte_carlo, filename)


def main_with_selective_output():
    """Main function that shows output on console AND saves to file"""
    doc_gen = TextDocumentGenerator()
    
    print("NFL Game Predictor - Output to Console and File")
    print("Results will show here AND be saved to file")
    print("=" * 60)
    
    # Get JSON file path
    from utils import get_json_file_path
    json_file_path = get_json_file_path()
    
    if not json_file_path:
        print("No data file provided. Exiting.")
        return
    
    print(f"Data file: {json_file_path}")
    print(f"Output directory: {doc_gen.output_dir}")
    
    print("\nChoose an option:")
    print("1. Regular season predictions")
    print("2. Single playoff simulation")
    print("3. Both regular season and playoff simulation")
    print("4. Monte Carlo playoff simulation")
    
    choice = input("Enter your choice (1-4): ").strip()
    
    if choice == "1":
        result, filename = doc_gen.run_regular_season_to_file(json_file_path)
        
    elif choice == "2":
        season_input = input("Enter season to simulate (or press Enter for latest): ").strip()
        season = int(season_input) if season_input else None
        result, filename = doc_gen.run_playoff_simulation_to_file(json_file_path, season)
        
    elif choice == "3":
        result, filename = doc_gen.run_both_predictions_to_file(json_file_path)
        
    elif choice == "4":
        season_input = input("Enter season to simulate (or press Enter for latest): ").strip()
        season = int(season_input) if season_input else None
        
        num_sims_input = input("Number of simulations (default 100): ").strip()
        num_sims = int(num_sims_input) if num_sims_input else 100
        
        result, filename = doc_gen.run_monte_carlo_to_file(json_file_path, season, num_sims)
        
    else:
        print("Invalid choice. Please run again and select 1-4.")
        return
    
    print(f"\nProcess completed!")
    print(f"Full report saved to: {os.path.abspath(filename)}")


if __name__ == "__main__":
    main_with_selective_output()