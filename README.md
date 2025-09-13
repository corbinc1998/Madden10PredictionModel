# Madden 10 Fantasy Draft Simulation

This repository contains a **Madden 10 Fantasy Draft Simulation**, a project that records every CPU vs. CPU game in a Madden 10 fantasy draft league. All game results are tracked and stored, and the project has now evolved to include an **AI predictor** that uses historical results to predict future outcomes. We are currently in **Season 6**, with the AI updating its model weekly based on real results.

All games are streamed live and uploaded to YouTube, and **results are updated in real time on the web app**.

---

## Links

- **YouTube Channel (Live and Archived Games):** [SimSportsStooge](https://www.youtube.com/@simsportsstooge/streams)  
- **Web App:** [The Sim League](https://thesimleague.web.app/)  
- **Predictions Log:** [View Detailed Predictions](https://github.com/corbinc1998/Madden10PredictionModel/blob/master/PredictionsLog.md)

---

## I AM CURRENTLY WORKING ON IMPLEMENTING TEAM STATS FROM ALL 5 SEASONS, I MAY REPREDICT

## AI Prediction For Week 2 (Current Week)

**Games Evaluated:** 16  
**Correct Predictions:** 12  
**Accuracy:** 75%  

### Full Week 2 Predictions (After Week 1 Adjustments)

- atl vs chi → **atl win** (50.3%)  
- cin vs car → **car win** (55.0%)  
- den vs ne → **den win** (61.3%)  
- det vs no → **det win** (61.5%)  
- gb vs buf → **gb win** (60.4%)  
- ind vs ten → **ind win** (56.3%)  
- mia vs nyj → **nyj win** (55.2%)  
- min vs ari → **ari win** (56.5%)  
- nyg vs jax → **nyg win** (53.4%)  
- oak vs kc → **oak win** (50.6%)  
- pit vs bal → **bal win** (57.1%)  
- sd vs hou → **sd win** (65.7%)  
- sea vs stl → **stl win** (54.8%)  
- sf vs dal → **dal win** (55.6%)  
- tb vs cle → **tb win** (65.0%)  
- was vs phi → **was win** (63.2%)  

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

