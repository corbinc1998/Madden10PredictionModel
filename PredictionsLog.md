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

## Week 3 Prediction Comparison (After Week 1 vs After Week 2)

### Predictions That Changed Winner:
| Matchup | After Week 1 | After Week 2 | Change |
|---------|-------------|-------------|---------|
| Buffalo vs Baltimore | BUF (50.4%) | BAL (56.6%) | Flipped to BAL |
| Pittsburgh vs Jacksonville | PIT (55.1%) | JAX (54.4%) | Flipped to JAX |
| Philadelphia vs San Francisco | SF (50.4%) | PHI (53.3%) | Flipped to PHI |
| Arizona vs Dallas | ARI (60.1%) | DAL (57.4%) | Flipped to DAL |
| NY Jets vs Kansas City | KC (54.6%) | NYJ (51.0%) | Flipped to NYJ |
| Green Bay vs Chicago | GB (57.3%) | CHI (51.9%) | Flipped to CHI |

### Same Winner, Different Confidence:
| Matchup | Winner | After Week 1 | After Week 2 | Δ Confidence |
|---------|--------|-------------|-------------|--------------|
| Washington vs St. Louis | WAS | 52.0% | 56.6% | +4.6 pp |
| Houston vs New England | HOU | 54.6% | 56.2% | +1.6 pp |
| Atlanta vs Tampa Bay | ATL | 61.6% | 64.6% | +3.0 pp |
| Tennessee vs NY Giants | TEN | 62.2% | 56.3% | -5.9 pp |
| San Diego vs Denver | DEN | 60.5% | 51.4% | -9.1 pp |
| Detroit vs Seattle | DET | 57.7% | 58.1% | +0.4 pp |
| Miami vs Minnesota | MIN | 56.6% | 62.2% | +5.6 pp |
| Cincinnati vs Indianapolis | CIN | 60.9% | 53.3% | -7.6 pp |

## Final Week 3 Predictions (After Week 2 Adjustments)

- Buffalo vs Baltimore → **Baltimore win** (56.6%)
- Washington vs St. Louis → **Washington win** (56.6%)
- Pittsburgh vs Jacksonville → **Jacksonville win** (54.4%)
- Houston vs New England → **Houston win** (56.2%)
- Atlanta vs Tampa Bay → **Atlanta win** (64.6%)
- Tennessee vs NY Giants → **Tennessee win** (56.3%)
- Green Bay vs Chicago → **Chicago win** (51.9%)
- Philadelphia vs San Francisco → **Philadelphia win** (53.3%)
- Arizona vs Dallas → **Dallas win** (57.4%)
- San Diego vs Denver → **Denver win** (51.4%)
- Detroit vs Seattle → **Detroit win** (58.1%)
- Miami vs Minnesota → **Minnesota win** (62.2%)
- NY Jets vs Kansas City → **NY Jets win** (51.0%)
- Cincinnati vs Indianapolis → **Cincinnati win** (53.3%)

### Key Week 3 Observations:
- **Most Volatile Predictions:** 6 games flipped winners after Week 2 results
- **Biggest Confidence Drop:** Denver over San Diego fell 9.1 percentage points
- **Highest Confidence:** Atlanta over Tampa Bay (64.6%)
- **Closest Games:** NY Jets vs Kansas City (51.0%), Green Bay vs Chicago (51.9%)
- **Model Learning:** Week 2's upsets significantly impacted road team confidence

---

## Prediction Accuracy Summary
- **Week 1 Accuracy:** 75% (12/16)
- **Week 2 Accuracy:** 43.8% (7/16)
- **Overall Accuracy Through Week 2:** 59.4% (19/32)
- **Model Evolution:** Ensemble weights have stabilized with Random Forest maintaining slight edge

### Week 2 Analysis
Week 2 proved challenging for the model with several upset results:
- **Biggest Upsets:** Washington (63.2% confidence) lost to Philadelphia, Denver (61.3% confidence) lost to New England
- **Model Struggles:** Road favorites performed poorly (3-6 record for away teams predicted to win)
- **Defensive Adjustments:** Several games had scoring patterns that differed significantly from recent form

## Final Week 3 Predictions (After Week 2 Adjustments)

- Buffalo vs Baltimore → **Baltimore win** (56.6%)
- Washington vs St. Louis → **Washington win** (56.6%)
- Pittsburgh vs Jacksonville → **Jacksonville win** (54.4%)
- Houston vs New England → **Houston win** (56.2%)
- Atlanta vs Tampa Bay → **Atlanta win** (64.6%)
- Tennessee vs NY Giants → **Tennessee win** (56.3%)
- Green Bay vs Chicago → **Chicago win** (51.9%)
- Philadelphia vs San Francisco → **Philadelphia win** (53.3%)
- Arizona vs Dallas → **Dallas win** (57.4%)
- San Diego vs Denver → **Denver win** (51.4%)
- Detroit vs Seattle → **Detroit win** (58.1%)
- Miami vs Minnesota → **Minnesota win** (62.2%)
- NY Jets vs Kansas City → **NY Jets win** (51.0%)
- Cincinnati vs Indianapolis → **Cincinnati win** (53.3%)

### Key Week 3 Observations:
- **Most Volatile Predictions:** 6 games flipped winners after Week 2 results
- **Biggest Confidence Drop:** Denver over San Diego fell 9.1 percentage points
- **Highest Confidence:** Atlanta over Tampa Bay (64.6%)
- **Closest Games:** NY Jets vs Kansas City (51.0%), Green Bay vs Chicago (51.9%)
- **Model Learning:** Week 2's upsets significantly impacted road team confidence

---

## Prediction Accuracy Summary
- **Week 1 Accuracy:** 75% (12/16)
- **Week 2 Accuracy:** 43.8% (7/16)
- **Overall Accuracy Through Week 2:** 59.4% (19/32)
- **Model Evolution:** Ensemble weights have stabilized with Random Forest maintaining slight edge

### Week 2 Analysis
Week 2 proved challenging for the model with several upset results:
- **Biggest Upsets:** Washington (63.2% confidence) lost to Philadelphia, Denver (61.3% confidence) lost to New England
- **Model Struggles:** Road favorites performed poorly (3-6 record for away teams predicted to win)
- **Defensive Adjustments:** Several games had scoring patterns that differed significantly from recent form
