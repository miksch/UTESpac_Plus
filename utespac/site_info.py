"""siteInfo – template for site-specific configuration (French Meadows / UU1).

Copy this file to ``<siteFolder>/siteInfo.py`` and adjust the values.
The main script executes it with ``exec()`` and reads the variables listed below.
"""

# Sonic orientations (azimuth from north, degrees).
# Order: tables sorted alphabetically, then columns in ascending height order.
sonicOrientation = [247, 120, 119]  # UU1

# Sonic manufacturer code per sonic:
#   1 = Campbell (CSAT3 / IRGASON)
#   0 = RMYoung  (v_RMY = u_Campbell !)
#   2 = Gill WindmasterPro
sonicManufact = [1, 1, 1]

# Orientation of the tower body relative to true north [degrees].
tower = 210

# Site elevation above sea level [m].
siteElevation = 1980

# Slope angle [degrees] – used for SNSP heat flux decomposition.
angle = 0.0

# Expected table names (must match the CSV filename table identifier).
tableNames = ["FM_20Hz_FluxTower"]

# Sampling frequency for each table [Hz].
tableScanFrequency = [20]

# Number of columns expected in each raw CSV table (before date compression).
tableNumberOfColumns = [19]

# Use local mean reference temperature from slow sensors (HMP) for virtual temperature.
useTrefHMP = True

# Averaging period [min] used to match fast (sonic) data to slow (HMP) data.
avgSlowFreq = 1

# Set True only when running a single high sonic (e.g. siteGill at 51.5 m) but needing
# virtual theta / humidity referenced to the lowest sonic on the full tower.
shiftzRef = False

# Override zRef value [m]; only used when shiftzRef = True.
zRefLowestSon = 4.42

# True if sonic heights in the input data table are in ascending order (low → high).
# Set False if heights are stored high → low (Diane's French Meadows tables).
ascending = True

# Sub-period length [min] for the SSITC steady-state test.  Must evenly divide avgPer.
SSITC_subAvgMin = 5

# Zero-plane displacement height [m] for SSITC stability parameter z/L.
displacementHeight = 0

# Canopy height [m]; used by the Rannik et al. in-canopy σ_w/u* ITC model.
# Set to float('nan') to disable the canopy branch.
canopyHeight = float('nan')

# If True and z ≤ canopyHeight, use the Rannik et al. canopy ITC model instead of
# the Foken above-canopy parameterization.
useCanopyITC = True
