"""Site configuration for siteVAC001 (converted from siteInfo.m)."""

sonicOrientation     = [215]  # UU1
sonicManufact        = [1]  # 1 IRGASONs
tower                = "001"  # VAC_001
siteElevation        = 18.2  # (m)
angle                = 0.  # mean of 20 m radius buffer
tableNames           = ["VAC001_20hz", "VAC001_30min"]  
tableScanFrequency   = [20, 1/(30*60)]  # [Hz]
tableNumberOfColumns = [48, 10]  

useTrefHMP      = True
avgSlowFreq     = 30
shiftsSonHeight = [10.85]   # sonic heights that have paired HMP sensors
shiftsHMPHeight = [10.72]   # corresponding physical HMP heights [m]
shiftzRef     = False
zRefLowestSon = 10.85
ascending     = False  # heights stored high → low in input tables

# SSITC quality flag settings
SSITC_subAvgMin   = 5     # sub-period length [min] for steady-state test
displacementHeight = 0    # zero-plane displacement height [m]
canopyHeight      = 6.0  # canopy height [m] for in-canopy ITC model
useCanopyITC      = True  # use Rannik et al. canopy σ_w/u* for z ≤ canopyHeight
