import json
import os

# Try to import tkinter for file dialog
try:
    import tkinter as tk
    from tkinter import filedialog
    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False
    print("Warning: tkinter not available. File dialog option will be disabled.")


def load_nfl_data(file_path):
    """Load NFL data from JSON file"""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: File not found - {file_path}")
    except json.JSONDecodeError:
        raise ValueError(f"Error: Invalid JSON file - {file_path}")
    except Exception as e:
        raise Exception(f"Error loading data: {e}")


def get_json_file_path():
    """Get JSON file path from user - either through file dialog or manual input"""
    print("\nHow would you like to load your NFL data?")
    print("1. Use file dialog to browse and select JSON file")
    print("2. Enter file path manually")
    print("3. Use default filename in current directory")
    
    choice = input("Enter your choice (1-3): ").strip()
    
    if choice == "1" and TKINTER_AVAILABLE:
        # Use file dialog
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        
        file_path = filedialog.askopenfilename(
            title="Select NFL JSON Data File",
            filetypes=[
                ("JSON files", "*.json"),
                ("All files", "*.*")
            ]
        )
        root.destroy()
        
        if not file_path:
            print("No file selected.")
            return None
        return file_path
        
    elif choice == "2":
        # Manual file path input
        file_path = input("Enter the full path to your JSON file: ").strip()
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return None
        return file_path
        
    elif choice == "3":
        # Check for default filenames in current directory
        default_names = [
            "nfl-standings-data.json",
            "nfl_data.json",
            "data.json"
        ]
        
        for filename in default_names:
            if os.path.exists(filename):
                print(f"Found file: {filename}")
                use_it = input(f"Use this file? (y/n): ").strip().lower()
                if use_it == 'y':
                    return filename
        
        print("No default files found in current directory.")
        return None
    
    else:
        if choice == "1" and not TKINTER_AVAILABLE:
            print("File dialog not available. tkinter is not installed.")
        else:
            print("Invalid choice.")
        return None


def display_predictions(predictions, season):
    """Display regular season predictions in a formatted way"""
    if predictions.empty:
        print("No upcoming games found to predict.")
        return False
    
    print(f"\nSeason {season} Game Predictions:")
    print("=" * 60)
    
    for _, pred in predictions.iterrows():
        winner = pred['predicted_winner']
        confidence = pred['confidence']
        
        if pred['home_team'] == winner:
            home_away = "Home"
        else:
            home_away = "Away"
            
        print(f"Week {pred['week']}: {pred['home_team']} vs {pred['away_team']}")
        print(f"  Predicted Winner: {winner} ({home_away})")
        print(f"  Confidence: {confidence:.1%}")
        print()
    
    return True


def display_simulation_summary(results):
    """Display playoff simulation summary"""
    if not results:
        print("No simulation results to display.")
        return
    
    print(f"\n{'='*25} FINAL SUMMARY {'='*25}")
    print(f"🏆 SUPER BOWL CHAMPION: {results['champion']}")
    print(f"🏈 Super Bowl Matchup: {results['super_bowl_teams'][0]} vs {results['super_bowl_teams'][1]}")
    
    # Determine conference champions
    afc_teams = [team for team in results['super_bowl_teams'] 
                if team in ['ne', 'buf', 'mia', 'nyj', 'pit', 'bal', 'cin', 'cle', 
                          'ind', 'jax', 'hou', 'ten', 'den', 'kc', 'oak', 'sd']]
    nfc_teams = [team for team in results['super_bowl_teams'] 
                if team in ['dal', 'nyg', 'phi', 'was', 'gb', 'chi', 'det', 'min',
                          'no', 'atl', 'car', 'tb', 'sf', 'sea', 'ari', 'stl']]
    
    if afc_teams and nfc_teams:
        print("\nConference Champions:")
        print(f"  AFC Champion: {afc_teams[0]}")
        print(f"  NFC Champion: {nfc_teams[0]}")


def create_test_data():
    """Create minimal test data for debugging"""
    return {
        "seasons": {
            "1": {
                "games": [
                    {
                        "week": 1,
                        "homeTeamId": "pit",
                        "awayTeamId": "ten",
                        "homeScore": 16,
                        "awayScore": 16,
                        "date": "2021-09-01T00:00:00",
                        "id": "7882013082776",
                        "completed": True
                    },
                    {
                        "week": 1,
                        "homeTeamId": "cin",
                        "awayTeamId": "den",
                        "homeScore": 24,
                        "awayScore": 35,
                        "date": "2021-09-01T00:00:00",
                        "id": "2468891558529",
                        "completed": True
                    }
                    # Note: In real usage, you'd need a full season of games
                ]
            }
        }
    }