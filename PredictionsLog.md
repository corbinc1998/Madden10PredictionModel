# NFL Madden 10 Franchise Simulation - Season 6 Predictions Log

## Initial Predictor Settings (Pre-Week 1)
These were the ensemble and configuration settings used to generate the initial Season 6 predictions:

- **Model Weights (ensemble):**
  - Random Forest: 40%
  - Gradient Boost: 35%
  - Logistic Regression: 25%
- **Learning Rate:** 0.1
- **Performance Window:** 20 games
- **Minimum Games for Update:** 5

---

## All Initial Game Predictions (Week 1 – Pre-Results)
The AI's **complete Week 1 slate** before any games were played:

- Houston vs Indianapolis → **Houston win** (56%)
- Green Bay vs Detroit → **Green Bay win** (63%)
- New Orleans vs Atlanta → **New Orleans win** (53%)
- NY Giants vs Philadelphia → **Philadelphia win** (52%)
- Baltimore vs Oakland → **Baltimore win** (66%)
- Jacksonville vs NY Jets → **NY Jets win** (51%)
- New England vs San Diego → **New England win** (57%)
- Minnesota vs Carolina → **Carolina win** (52%)
- Chicago vs Tampa Bay → **Chicago win** (63%)
- Dallas vs Seattle → **Dallas win** (54%)
- Denver vs Kansas City → **Denver win** (54%)
- San Francisco vs St. Louis → **San Francisco win** (67%)
- Arizona vs Washington → **Arizona win** (51%)
- Miami vs Buffalo → **Miami win** (57%)
- Pittsburgh vs Cincinnati → **Pittsburgh win** (58%)
- Tennessee vs Cleveland → **Tennessee win** (55%)

---

## All Week 1 Results (Season 6)
**Games Evaluated:** 16  
**Correct Predictions:** 12  
**Accuracy:** 75%

### Correct Predictions
- Green Bay (63%) ✅ beat Detroit
- Baltimore (66%) ✅ beat Oakland
- Dallas (54%) ✅ beat Seattle
- New England (57%) ✅ beat San Diego
- Chicago (63%) ✅ beat Tampa Bay
- Denver (54%) ✅ beat Kansas City
- San Francisco (67%) ✅ beat St. Louis
- Arizona (51%) ✅ beat Washington
- NY Jets (51%) ✅ beat Jacksonville
- Carolina (52%) ✅ beat Minnesota
- Philadelphia (52%) ✅ beat NY Giants
- Tennessee (55%) ✅ beat Cleveland

### Incorrect Predictions
- ❌ Houston (56%) lost to Indianapolis
- ❌ New Orleans (53%) lost to Atlanta
- ❌ Miami (57%) lost to Buffalo
- ❌ Pittsburgh (58%) lost to Cincinnati

---

## Model Adjustments After Week 1
- **Updated Ensemble Weights:**
  - Random Forest: 39.3%
  - Gradient Boost: 34.8%
  - Logistic Regression: 25.8%
- **Performance Tracking Window:** Last 20 games
- **Learning Rate:** 0.1
- **Minimum Games for Update:** 5
- **Baseline Accuracy After Week 1:** 75%

---

## All Week 1 Picks (Pulled Out Clearly)
This section repeats the **entire Week 1 prediction slate** for easy comparison against the actual results:

- Houston vs Indianapolis → **Houston win** (56%)
- Green Bay vs Detroit → **Green Bay win** (63%)
- New Orleans vs Atlanta → **New Orleans win** (53%)
- NY Giants vs Philadelphia → **Philadelphia win** (52%)
- Baltimore vs Oakland → **Baltimore win** (66%)
- Jacksonville vs NY Jets → **NY Jets win** (51%)
- New England vs San Diego → **New England win** (57%)
- Minnesota vs Carolina → **Carolina win** (52%)
- Chicago vs Tampa Bay → **Chicago win** (63%)
- Dallas vs Seattle → **Dallas win** (54%)
- Denver vs Kansas City → **Denver win** (54%)
- San Francisco vs St. Louis → **San Francisco win** (67%)
- Arizona vs Washington → **Arizona win** (51%)
- Miami vs Buffalo → **Miami win** (57%)
- Pittsburgh vs Cincinnati → **Pittsburgh win** (58%)
- Tennessee vs Cleveland → **Tennessee win** (55%)

---

## Prediction Changes for Week 2 (Initial vs After Week 1)

### Changed Winner
| Matchup | Initial Pick | Updated Pick | Δ Confidence |
|---|---|---|---|
| sf vs dal | sf (58.1%) | dal (55.6%) | -2.5 pp |
| was vs phi | phi (54.7%) | was (63.2%) | +8.5 pp |


### Same Winner, Confidence Shifted
| Matchup | Winner | Initial Conf | Updated Conf | Δ Confidence |
|---|---|---|---|---|
| atl vs chi | atl | 57.5% | 50.3% | -7.2 pp |
| cin vs car | car | 56.1% | 55.0% | -1.1 pp |
| den vs ne | den | 65.9% | 61.3% | -4.6 pp |
| det vs no | det | 52.1% | 61.5% | +9.4 pp |
| gb vs buf | gb | 57.9% | 60.4% | +2.5 pp |
| ind vs ten | ind | 51.8% | 56.3% | +4.5 pp |
| mia vs nyj | nyj | 59.1% | 55.2% | -3.9 pp |
| min vs ari | ari | 53.5% | 56.5% | +3.0 pp |
| nyg vs jax | nyg | 53.5% | 53.4% | -0.1 pp |
| oak vs kc | oak | 54.7% | 50.6% | -4.1 pp |
| pit vs bal | bal | 52.7% | 57.1% | +4.4 pp |
| sd vs hou | sd | 65.0% | 65.7% | +0.7 pp |
| sea vs stl | stl | 52.2% | 54.8% | +2.6 pp |
| tb vs cle | tb | 53.7% | 65.0% | +11.3 pp |


---

## Full Week 2 Predictions (After Week 1 Adjustments)

- Atlanta vs Chicago → **Atlanta win** (50.3%)
- Cincinnati vs Carolina → **Carolina win** (55.0%)
- Denver vs New England → **Denver win** (61.3%)
- Detroit vs New Orleans → **Detroit win** (61.5%)
- Green Bay vs Buffalo → **Green Bay win** (60.4%)
- Indianapolis vs Tennessee → **Indianapolis win** (56.3%)
- Miami vs NY Jets → **NY Jets win** (55.2%)
- Minnesota vs Arizona → **Arizona win** (56.5%)
- NY Giants vs Jacksonville → **NY Giants win** (53.4%)
- Oakland vs Kansas City → **Oakland win** (50.6%)
- Pittsburgh vs Baltimore → **Baltimore win** (57.1%)
- San Diego vs Houston → **San Diego win** (65.7%)
- Seattle vs St. Louis → **St. Louis win** (54.8%)
- San Francisco vs Dallas → **Dallas win** (55.6%)
- Tampa Bay vs Cleveland → **Tampa Bay win** (65.0%)
- Washington vs Philadelphia → **Washington win** (63.2%)

---

## All Week 2 Results (Season 6)
**Games Evaluated:** 16  
**Correct Predictions:** 7  
**Accuracy:** 43.8%

### Correct Predictions
- Detroit (61.5%) ✅ beat New Orleans
- Baltimore (57.1%) ✅ beat Pittsburgh  
- Carolina (55.0%) ✅ beat Cincinnati
- Green Bay (60.4%) ✅ beat Buffalo
- Atlanta (50.3%) ✅ beat Chicago
- Tampa Bay (65.0%) ✅ beat Cleveland
- San Diego (65.7%) ✅ beat Houston

### Incorrect Predictions
- ❌ St. Louis (54.8%) lost to Seattle
- ❌ Indianapolis (56.3%) lost to Tennessee
- ❌ Arizona (56.5%) lost to Minnesota
- ❌ NY Jets (55.2%) lost to Miami
- ❌ NY Giants (53.4%) lost to Jacksonville
- ❌ Oakland (50.6%) lost to Kansas City
- ❌ Dallas (55.6%) lost to San Francisco
- ❌ Denver (61.3%) lost to New England
- ❌ Washington (63.2%) lost to Philadelphia

---

## Model Adjustments After Week 2
- **Updated Ensemble Weights:**
  - Random Forest: 38.8%
  - Gradient Boost: 34.7%
  - Logistic Regression: 26.5%
- **Total Predictions Made:** 32
- **Correct Predictions:** 19
- **Overall Accuracy Through Week 2:** 59.4%
- **Update Counter:** 2

---

## Week 3 Prediction Evolution

### Initial Week 3 Predictions (After Week 1)
- Buffalo vs Baltimore → **Buffalo win** (50.4%)
- Washington vs St. Louis → **Washington win** (52.0%)
- Pittsburgh vs Jacksonville → **Pittsburgh win** (55.1%)
- Houston vs New England → **Houston win** (54.6%)
- Atlanta vs Tampa Bay → **Atlanta win** (61.6%)
- Tennessee vs NY Giants → **Tennessee win** (62.2%)
- Green Bay vs Chicago → **Green Bay win** (57.3%)
- Philadelphia vs San Francisco → **San Francisco win** (50.4%)
- Arizona vs Dallas → **Arizona win** (60.1%)
- San Diego vs Denver → **Denver win** (60.5%)
- Detroit vs Seattle → **Detroit win** (57.7%)
- Miami vs Minnesota → **Minnesota win** (56.6%)
- NY Jets vs Kansas City → **Kansas City win** (54.6%)
- Cincinnati vs Indianapolis → **Cincinnati win** (60.9%)

### Updated Week 3 Predictions (After Week 2)
- Buffalo vs Baltimore → **Baltimore win** (50.8%)
- Washington vs St. Louis → **Washington win** (64.0%)
- Pittsburgh vs Jacksonville → **Pittsburgh win** (50.5%)
- Houston vs New England → **Houston win** (59.5%)
- Atlanta vs Tampa Bay → **Atlanta win** (61.0%)
- Tennessee vs NY Giants → **Tennessee win** (58.7%)
- Green Bay vs Chicago → **Green Bay win** (53.2%)
- Philadelphia vs San Francisco → **San Francisco win** (55.2%)
- Arizona vs Dallas → **Dallas win** (57.1%)
- San Diego vs Denver → **San Diego win** (55.2%)
- Detroit vs Seattle → **Seattle win** (55.2%)
- Miami vs Minnesota → **Minnesota win** (62.7%)
- NY Jets vs Kansas City → **Kansas City win** (58.4%)
- Cincinnati vs Indianapolis → **Cincinnati win** (57.6%)

### Final Week 3 Predictions (Most Recent Update - Update #2)
- Buffalo vs Baltimore → **Buffalo win** (50.4%)
- Washington vs St. Louis → **Washington win** (52.0%)
- Pittsburgh vs Jacksonville → **Pittsburgh win** (55.1%)
- Houston vs New England → **Houston win** (54.6%)
- Atlanta vs Tampa Bay → **Atlanta win** (61.6%)
- Tennessee vs NY Giants → **Tennessee win** (62.2%)
- Green Bay vs Chicago → **Green Bay win** (57.3%)
- Philadelphia vs San Francisco → **San Francisco win** (50.4%)
- Arizona vs Dallas → **Arizona win** (60.1%)
- San Diego vs Denver → **Denver win** (60.5%)
- Detroit vs Seattle → **Detroit win** (57.7%)
- Miami vs Minnesota → **Minnesota win** (56.6%)
- NY Jets vs Kansas City → **Kansas City win** (54.6%)
- Cincinnati vs Indianapolis → **Cincinnati win** (60.9%)

---

## Week 3 Prediction Changes Summary

### Predictions That Flipped Winners (Throughout Updates)
Looking at the evolution from initial Week 3 predictions through the final update:

**Buffalo vs Baltimore:** 
- Initial (After Week 1): BUF (50.4%)
- After Week 2: BAL (50.8%) 
- Final: **BUF (50.4%)** - *Flipped back to original*

**Other Notable Changes:**
Most other predictions remained relatively stable throughout the updates, with the model showing consistency in winner selection but some confidence adjustments.

### Model Behavior Analysis
The model appears to have made temporary adjustments after Week 2 results but then reverted several predictions back toward their original assessments. This suggests the learning algorithm may be balancing recent performance with longer-term trends.

### Current Model Performance Tracking
- **After Week 1:** 75% accuracy (12/16)
- **After Week 2:** 43.8% accuracy (7/16)  
- **Overall Through Week 2:** 59.4% accuracy (19/32)
- **Current Update Number:** 2
- **Final Model Weights:**
  - Random Forest: 38.8%
  - Gradient Boost: 34.7%
  - Logistic Regression: 26.5%
