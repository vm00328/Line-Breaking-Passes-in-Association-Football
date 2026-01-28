# Beyond Pass Completion  
### Automated Detection of Line-Breaking Passes & the Pass Advantage Score (PAS)

This repository contains the implementation of a data-driven system for **automatically detecting line-breaking passes in association football** and introducing a novel metric - the **Pass Advantage Score (PAS)** for evaluating the true offensive impact of passes.

The project combines **tracking data** with **event data** and applies **unsupervised learning** to dynamically model defensive formation lines, enabling a richer analysis of passing performance beyond traditional pass completion metrics.

---

## 📌 Project Motivation

In modern football analytics, passes are often reduced to a binary outcome: *successful* or *unsuccessful*. This oversimplification ignores the **tactical value** of penetrative passes that break opposition lines and directly contribute to goal-scoring opportunities.

This project addresses that gap by:
- Automatically identifying **line-breaking passes**
- Quantifying their **strategic advantage**
- Enabling **scalable, objective post-match analysis**

The system was developed and validated using **Dutch Eredivisie match data**, in collaboration with **AZ Alkmaar**.

---

## 🧠 Core Concepts

### Line-Breaking Pass
A pass is classified as *line-breaking* if it:
- Is successful and from open play  
- Advances the ball by **at least 10% of the remaining passable area** toward the opponent’s goal  
- Is directed toward goal within a predefined angular threshold  
- **Intersects at least one formation line** of the opposing team  

Formation lines are inferred dynamically using **constrained K-Means clustering** on outfield player positions.

---

### Pass Advantage Score (PAS)
**PAS** quantifies the *quality* of a line-breaking pass by evaluating:
- The zone in which the pass is received  
- The proximity of the nearest defender to the receiver  

Passes received in **dynamic hot zones** (high-space, high-threat areas) receive higher PAS values, correlating with increased goal probability.

---

## ⚙️ System Pipeline

1. **Data Synchronization**  
   Alignment between manually annotated event data and high-frequency tracking data (25fps)

2. **Preprocessing**  
   - Pitch normalization (105 x 68 meters)  
   - Goalkeeper exclusion  
   - Ball-in-play filtering  

3. **Formation Line Detection**  
   - Constrained K-Means clustering on out-of-possession players  
   - Dynamic formation reconstruction per pass event  

4. **Line-Breaking Detection**  
   - Ball trajectory intersection with formation lines  
   - Tactical filtering based on domain constraints  

5. **Metric Computation**  
   - Pass Advantage Score (PAS)  
   - Visualization and player-level aggregation  

---

## 📊 Key Results

- Line-breaking passes show **~2x higher goal probability** compared to non-line-breaking passes  
- Average clustering quality:  
  - **Silhouette Score:** ~0.81  
  - **Davies–Bouldin Index:** ~0.19  
- Processing time reduced from **~225 minutes to ~2.5 minutes per match**  
- PAS effectively distinguishes high-impact passes and players with superior tactical vision  

---

## 🛠️ Tech Stack

- **Python**
- **NumPy / Pandas**
- **Scikit-learn**
- **mplsoccer**
- **Matplotlib / Seaborn**
- Tracking data: **TRACAB**
- Event data: **StatsBomb API**
