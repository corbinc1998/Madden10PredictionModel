# NFL Game Outcome Prediction Model

A machine learning model that predicts NFL game outcomes using historical franchise simulation data from Madden or similar football games.

## Overview

This model analyzes team performance patterns, momentum, and historical matchups to predict the outcome of scheduled games in an ongoing season. It automatically detects which season is currently in progress and provides win probabilities for remaining games.

## Quick Start

### 1. Install Dependencies
```bash
pip install pandas numpy scikit-learn
```

### 2. Run the Predictor
```bash
python prediction_model.py
```

### 3. Choose How to Load Your Data
When you run the script, you'll see these options:
```
How would you like to load your NFL data?
1. Use file dialog to browse and select JSON file
2. Enter file path manually  
3. Use default filename in current directory
```

- **Option 1**: Opens Finder (Mac) or File Explorer (Windows) to browse for your JSON file
- **Option 2**: Type the full path to your JSON file
- **Option 3**: Automatically finds common filenames like `nfl_data.json` in the current directory

## Data Format

The model expects JSON data exported from a franchise simulation with the following structure:

### Top Level Structure
```json
{
  "seasons": {
    "1": { "games": [...], "name": "Season 1" },
    "2": { "games": [...], "name": "Season 2" },
    ...
  },
  "exportDate": "2025-08-25T11:57:56.652Z"
}
```

### Game Object Structure
Each game should have the following fields:

**Completed Games:**
```json
{
  "week": 1,
  "homeTeamId": "pit",
  "awayTeamId": "cin",
  "homeScore": 16,
  "awayScore": 38,
  "date": "2025-07-15T00:00:00.000Z",
  "id": "1752598266948",
  "completed": true,
  "status": "final"
}
```

**Scheduled Games (to be predicted):**
```json
{
  "week": 1,
  "homeTeamId": "hou",
  "awayTeamId": "ind",
  "date": "2025-08-25T00:00:00.000Z",
  "status": "scheduled",
  "homeScore": null,
  "awayScore": null,
  "id": "1756120446699"
}
```

**Playoff Games:**
```json
{
  "round": "wildcard",
  "conference": "AFC",
  "matchup": 1,
  "homeTeamId": "cle",
  "awayTeamId": "ind",
  "homeScore": 13,
  "awayScore": 31,
  "isPlayoff": true,
  "week": 18,
  "seasonId": "5",
  "completed": true
}
```

### Team ID Format
Teams are identified using standard 2-3 letter abbreviations:
- `pit` (Pittsburgh)
- `ne` (New England) 
- `gb` (Green Bay)
- `sf` (San Francisco)
- `sd` (San Diego)
- `no` (New Orleans)
- etc.

## Sample Data

Here's a minimal example with one complete season and one in-progress season:

```json
{
  "seasons": {
    "1": {
      "games": [
        {
          "week": 1,
          "homeTeamId": "pit",
          "awayTeamId": "cin",
          "homeScore": 16,
          "awayScore": 38,
          "date": "2024-09-01T00:00:00",
          "id": "game_1",
          "completed": true
        },
        {
          "week": 1,
          "homeTeamId": "dal",
          "awayTeamId": "gb",
          "homeScore": 24,
          "awayScore": 21,
          "date": "2024-09-01T00:00:00",
          "id": "game_2", 
          "completed": true
        }
      ],
      "name": "Season 1"
    },
    "2": {
      "games": [
        {
          "week": 1,
          "homeTeamId": "ne",
          "awayTeamId": "buf",
          "homeScore": 28,
          "awayScore": 24,
          "date": "2025-09-01T00:00:00",
          "id": "game_3",
          "completed": true
        },
        {
          "week": 2,
          "homeTeamId": "mia",
          "awayTeamId": "nyj",
          "date": "2025-09-08T00:00:00",
          "status": "scheduled",
          "homeScore": null,
          "awayScore": null,
          "id": "game_4"
        }
      ],
      "name": "Season 2"
    }
  },
  "exportDate": "2025-09-05T12:00:00.000Z"
}
```

## Features Used for Prediction

The model analyzes these factors to make predictions:

### Team Performance Metrics
- **Recent Form**: Last 5 games (wins, points scored, points allowed)
- **Season Performance**: Current season stats up to the prediction point
- **Point Differentials**: Offensive and defensive efficiency trends
- **Win Differentials**: Recent momentum between teams

### Contextual Factors
- **Head-to-Head History**: Historical matchups between the two teams
- **Home Field Advantage**: Built into the model structure
- **Playoff Context**: Whether the game is a playoff game

### Derived Features
- Points per game differential between teams
- Points allowed per game differential
- Recent wins differential
- Season wins differential

## Model Architecture

The prediction system uses an ensemble of three models:

1. **Random Forest**: Handles non-linear relationships and feature interactions
2. **Gradient Boosting**: Excellent for sequential patterns in team performance
3. **Logistic Regression**: Provides calibrated probability estimates

Final predictions are averaged across all three models for improved accuracy.

## Sample Output

When you run the predictor, you'll see output like this:

```
Processing game data...
Processed 256 completed games across 5 seasons

Calculating team statistics...

Training prediction models...
random_forest: CV Accuracy = 0.625 (+/- 0.089)
gradient_boost: CV Accuracy = 0.609 (+/- 0.095)
logistic_regression: CV Accuracy = 0.594 (+/- 0.102)

Top 10 Most Important Features (Random Forest):
           feature  importance
    ppg_differential    0.125432
   recent_win_diff     0.098765
  home_recent_ppg      0.087432
  away_recent_papg     0.076543

Predicting upcoming games...

Season 6 Game Predictions:
============================================================
Week 10: hou vs ind
  Predicted Winner: ind (Away)
  Confidence: 68.5%

Week 10: pit vs cin
  Predicted Winner: pit (Home)
  Confidence: 72.3%
```

## Installation Options

### Option 1: Simple Installation (Recommended)
```bash
pip install pandas numpy scikit-learn
python prediction_model.py
```

### Option 2: Virtual Environment (Advanced)
```bash
# Create virtual environment
python -m venv nfl_predictor
cd nfl_predictor

# Activate virtual environment
# On Windows:
Scripts\activate
# On macOS/Linux:
source bin/activate

# Install dependencies
pip install pandas numpy scikit-learn

# Run the predictor
python prediction_model.py

# Deactivate when done
deactivate
```

## Requirements

Main dependencies:
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical operations and arrays
- **scikit-learn**: Machine learning models and preprocessing
- **tkinter**: File dialog (usually included with Python)

## Programmatic Usage

If you want to use the predictor in your own code:

```python
from prediction_model import NFLGamePredictor

# Initialize with JSON file
predictor = NFLGamePredictor('your_franchise_data.json')

# Or with data dictionary
predictor = NFLGamePredictor(data_dict=your_data)

# Process and train
predictor.process_data()
predictor.calculate_team_stats()
predictor.train_models()

# Get predictions for upcoming games
predictions = predictor.predict_latest_season_games()

# Access predictions
for _, pred in predictions.iterrows():
    print(f"{pred['home_team']} vs {pred['away_team']}")
    print(f"Winner: {pred['predicted_winner']}")
    print(f"Confidence: {pred['confidence']:.1%}")
```

### Output Format
The model returns a pandas DataFrame with these columns:
- `week`: Game week number
- `home_team`: Home team abbreviation
- `away_team`: Away team abbreviation  
- `home_win_probability`: Probability home team wins (0-1)
- `predicted_winner`: Predicted winning team
- `confidence`: Confidence level (0.5-1.0)

## Troubleshooting

### Common Issues

**"No file selected" or file dialog doesn't open:**
- Try Option 2 (manual file path input)
- Make sure tkinter is installed: `pip install tk`

**"Invalid JSON file" error:**
- Verify your JSON file is properly formatted
- Check that the file contains the required `seasons` structure

**"Missing required data" error:**
- Ensure completed games have `homeScore`, `awayScore`, and `completed: true`
- Verify scheduled games have `status: "scheduled"`

**Low prediction accuracy:**
- Model works best with 3+ complete seasons of training data
- Ensure your data includes a variety of teams and matchups

### File Path Examples

**Windows:**
```
C:\Users\YourName\Documents\nfl_data.json
```

**Mac/Linux:**
```
/Users/YourName/Documents/nfl_data.json
```

## Limitations

- Requires at least one complete season for training
- Predictions are based on team-level statistics only
- Does not account for injuries, weather, or other external factors
- Best performance with 3+ complete seasons of training data
- Accuracy typically ranges from 55-70% depending on data quality

## Contributing

This model can be extended with additional features such as:
- Strength of schedule calculations
- Player-level performance metrics
- Weather and venue factors
- Coaching and staff changes
- More sophisticated ensemble methods
- Real-time data integration
- Advanced feature engineering

## License

This project is open source. Feel free to modify and distribute according to your needs.