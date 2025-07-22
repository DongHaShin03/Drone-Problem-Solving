# Drone Inspection Route Planner

## Problem Description

A company with multiple buildings across the country uses drones to inspect the external surfaces of each site using cameras and various sensors.  
For each building, a set of **3D coordinates** is provided where the drone must **stop** to collect data.  
The **orientation** of the drone at each point can be neglected.

The drone:

- Starts from a **base location** `(x0, y0, z0)`, chosen from a predefined set
- Visits a **subset of points**, depending on its battery level
- Returns to the **base for recharging**
- Repeats the process until **all points are covered**, possibly over multiple trips

---

## Drone Movement

The drone moves at different speeds based on direction:

- `1 m/s` for **ascending**
- `2 m/s` for **descending**
- `1.5 m/s` for **horizontal** movement
- For **oblique movements**:  
  If the segment has horizontal length `a` and vertical ascent `b`,  
  the time is calculated as:

  max { a / 1.5, b / 1.0 } seconds


> Acceleration and deceleration can be ignored.

---

## Energy Consumption

Battery usage depends on direction:

- `50 J/m` for **ascending**
- `5 J/m` for **descending**
- `10 J/m` for **horizontal**
- For **oblique movement**, the energy is the **sum** of the vertical and horizontal components

---

## Objectives

Determine:

1. The **base station** position
2. The **drone routes** for all trips

Such that:

- All grid points (including entry points) are **visited at least once**
- The drone **returns to base** when necessary to recharge
- The **total flight time is minimized**

---

## Connectivity Rules

Two grid points **A** and **B** are **connected** if:

- The **Euclidean distance** between A and B is **≤ 4 meters**,  
**OR**
- The distance is **≤ 11 meters**, and at least **two coordinates (x, y, or z)** differ by at most **0.5 meters**

These rules **do not apply** for segments between the base and the **grid entry points**:  
The drone can freely travel between the base and any **entry point**, regardless of distance.

---

## Dataset Description

Two CSV files contain the 3D points for two buildings:

### 🏢 `Edificio1.csv`

- Each row: a point `(x, y, z)` to inspect
- **Base candidates**: integer points with:
- `-8 ≤ x ≤ 5`
- `-17 ≤ y ≤ -15`
- `z = 0`
- **Entry points**: all points with `y ≤ -12.5`
- **Battery capacity**: `1 Wh`

---

### `Edificio2.csv`

- Each row: a point `(x, y, z)` to inspect
- **Base candidates**: integer points with:
- `-10 ≤ x ≤ 10`
- `-31 ≤ y ≤ -30`
- `z = 0`
- **Entry points**: all points with `y ≤ -20`
- **Battery capacity**: `6 Wh`

---



  

