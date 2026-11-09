# Instrument specifications: CSAT3, LI-7500DS, IRGASON with EC100

What the three eddy-covariance heads record, at which rate, with which
filter and delay, and how each flags a bad sample, read from the
manufacturers' manuals. This is the instrument note that
[sonic_temperature_flux.md](sonic_temperature_flux.md),
[ec_spectra.md](ec_spectra.md) and the flux engine's lag and quality
tests rest on. Sources, each read from its extraction under
`library/extracted/` on 2026-11-05:

- CSAT3 Three Dimensional Sonic Anemometer, instruction manual,
  Revision 4/17, Campbell Scientific [@CampbellCSAT3]
  (`csat3.txt`). Printed page = PDF page - 10 in the body `[DERIVED]`
  from page headers.
- Using the LI-7500DS Open Path CO2/H2O Analyzer and the SmartFlux 3
  System, publication 984-16770, LI-COR [@LICOR7500DS]
  (`7500DS_Manual_16770.txt`). Printed pages are section-numbered
  (5-2, 10-13, B-1).
- IRGASON Integrated CO2/H2O Open-Path Gas Analyzer and 3D Sonic
  Anemometer, instruction manual, Revision 8/17, Campbell Scientific
  [@CampbellIRGASON] (`irgason_man.txt`). Printed page = PDF page - 11
  in the body `[DERIVED]` from page headers.

Each manual is written for its own head, in any deployment; none names
a campaign `[DERIVED]`. Every row below is quoted from the text layer of the
manual (no value is read from a page image). Loci give section,
printed page and PDF page; quotes are written in ASCII: curly quotes
and apostrophes as straight ones, ligatures spelled out, en-dash and
minus sign as "-", degree sign as "deg", plus-minus as "+/-", micro
sign as "u", gamma as "gamma", multiplication sign as "x", asterisk
operator as "*", middle dot as "."; footnote digits such as
"bandwidth1" are kept. " [...] " joins two passages of the same page
`[HOUSE]`. Which head, embedded code, execution parameter and
bandwidth a given tower ran is not established by any manual; it sits
in the logger program and the site records `[DERIVED]`.

## CSAT3 -- sonic anemometer [@CampbellCSAT3]

| ID | Quantity | Value | Quote | Locus | Label |
|---|---|---|---|---|---|
| C1 | Path length | 10.0 cm vertical, 5.8 cm horizontal | "MEASUREMENT PATH LENGTH: 10.0 cm vertical; 5.8 cm horizontal" | Sec 2.3, p. 3, PDF 13 | `[CITED]` |
| C2 | Path angle | 60 deg from horizontal | "TRANSDUCER PATH ANGLE FROM HORIZONTAL: 60 degrees" | Sec 2.3, p. 3, PDF 13 | `[CITED]` |
| C3 | Measurement rate | 1 to 60 Hz single; oversampled modes block-averaged to 10 or 20 Hz | "MEASUREMENT RATE: programmable from 1 to 60 Hz, instantaneous measurements; two oversampled modes are block averaged to either 20 Hz or 10 Hz" | Sec 2.1, p. 2, PDF 12 | `[CITED]` |
| C4 | Single-measurement mode | one measurement per trigger | "In the single-measurement mode, the anemometer makes one measurement per trigger." | Sec 8.1.1, p. 18, PDF 28 | `[CITED]` |
| C5 | Single-measurement response | geometry-limited | "the anemometer's high frequency response is limited only by the geometry of the anemometer head" | Sec 8.1.1, p. 18, PDF 28 | `[CITED]` |
| C6 | Aliasing, single mode | aliases | "this approach also aliases high-frequency information to lower frequencies." | Sec 8.1.1, p. 18, PDF 28 | `[CITED]` |
| C7 | Aliasing and fluxes | no effect on variances and covariances | "This aliasing does not compromise the variances and covariances (and therefore, fluxes) computed from aliased data." | Sec 8.1.1, p. 18, PDF 28 | `[CITED]` |
| C8 | Oversample mode | 60 Hz block-averaged to 10 or 20 Hz | "In the oversample mode, the CSAT3 makes wind measurements at 60 Hz and then block averages those measurements to 10 or 20 Hz output." | Sec 8.1.2, p. 18, PDF 28 | `[CITED]` |
| C9 | Oversample filter | sin x/x | "The 60 Hz data is filtered with a sin x/x function." | Sec 8.1.2, p. 18, PDF 28 | `[CITED]` |
| C10 | Oversample effect | limits high-frequency response | "The oversampling modes minimize aliasing by limiting the anemometer's high-frequency response." | Sec 8.1.2, p. 18, PDF 28 | `[CITED]` |
| C11 | Execution parameter, 60 Hz oversample | options 61 (10 Hz) and 62 (20 Hz) | "the wind measurements at a rate of 60 Hz and expect a trigger at a rate of 10 Hz (option 61) or 20 Hz (option 62)" | Sec 10.3, p. 21, PDF 31 | `[CITED]` |
| C12 | Execution parameter, 48 Hz oversample | options 51 (8 Hz) and 52 (16 Hz) | "at a rate of 48 Hz and expect a trigger at a rate of 8 Hz (option 51) or 16 Hz (option 52)" | Sec 10.3, p. 21, PDF 31 | `[CITED]` |
| C13 | Pipeline delay, summary | two scans | "All the output signals have a two scan delay." | Sec 2.2, p. 2, PDF 12 | `[CITED]` |
| C14 | Pipeline delay, all outputs | two measurements | "All the CSAT3 output signals (SDM, RS-232, and analog) have the two measurement pipeline delay." | Sec 9, p. 18, PDF 28 | `[CITED]` |
| C15 | Pipeline delay, meaning | data measured two triggers before output | "were measured by the anemometer two triggers ago" | Sec 9, p. 18, PDF 28 | `[CITED]` |
| C16 | Pipeline delay, rate dependence | applies at every trigger rate | "This delay applies at all trigger frequencies." | Sec 9, p. 18, PDF 28 | `[CITED]` |
| C17 | Oversampled delay | two master-trigger periods | "block-averaged output data have a fixed delay of two periods of the master trigger." | Sec 8.1.2, p. 18, PDF 28 | `[CITED]` |
| C18 | Alignment with other sensors | user's task, online in the logger | "the data from each sensor has to be aligned in time. This can be done online with a datalogger." | Sec 9, p. 18, PDF 28 | `[CITED]` |
| C19 | Ts definition | mean of the three non-orthogonal paths | "the reported speed of sound (c) or sonic virtual temperature (Ts), is the average between the three non-orthogonal sonic axes." | Sec 2.1, p. 1, PDF 11 | `[CITED]` |
| C20 | Crosswind correction | online, per path, before averaging | "corrects each of the three sonic paths speed of sound for the effects of wind blowing perpendicular to the sonic path. The three corrected speed of sound values are then averaged." | Sec 9, p. 18, PDF 28 | `[CITED]` |
| C21 | Crosswind correction, spec | applied in the head | "SPEED OF SOUND: determined from 3 acoustic paths; corrected for crosswind effects" | Sec 2.1, p. 2, PDF 12 | `[CITED]` |
| C22 | Crosswind, off-line | not to be repeated | "CSAT3 users need not make these corrections to their fluxes because the CSAT3 performs an online correction. Correcting CSAT3 data off-line for cross wind effects will cause errors in the measured fluxes." | Sec 7.3, p. 15, PDF 25 | `[CITED]` |
| C23 | Crosswind, appendix | no off-line correction required | "The CSAT3 corrects online for the effect of wind blowing perpendicular to the sonic path. No additional off-line corrections are required as suggested by Liu et al., 2001." | App C, p. C-1, PDF 65 | `[CITED]` |
| C24 | Ts equation constants | gamma_d = 1.4, R_d = 287.04 J K-1 kg-1 | "is given by Eq. (9), where gammad = 1.4 and Rd = 287.04 JK-1 kg-1" | App C, p. C-2, PDF 66 | `[CITED]` |
| C25 | Resolution | ux, uy 1 mm/s; uz 0.5 mm/s; c 15 mm/s (code v4), 1 mm/s (code v3) | "MEASUREMENT RESOLUTION: ux and uy are 1 mm s-1 rms; uz is 0.5 mm s-1 rms; c is 15 mm s-1 (0.025 degC) with embedded code version 4 (standard) [c is 1 mm s-1 (0.002 degC) with embedded code version 3]" | Sec 2.1, p. 2, PDF 12 | `[CITED]` |
| C26 | Resolution, code v4 against v3 | speed of sound 15 times coarser | "using embedded code version 4 is 15 times worse than embedded code version 3 (0.002degC to 0.025degC RMS)" | Sec 7.1, p. 14, PDF 24 | `[CITED]` |
| C27 | Operating temperature | -30 to 50 C standard; -40 to 40 C cold-shifted | "OPERATING TEMPERATURE RANGE: -30 to 50 degC (standard); -40 to 40 degC (cold shifted)" | Sec 2.1, p. 2, PDF 12 | `[CITED]` |
| C28 | Wind output range | +/-65.535 m/s | "Full scale wind: +/-65.535 m s-1 autoranging between four ranges; least significant bit is 0.25 to 2 mm s-1" | Sec 2.2, p. 3, PDF 13 | `[CITED]` |
| C29 | Speed-of-sound output range | 300 to 366 m/s (-50 to +60 C) | "Speed of sound: 300 to 366 m s-1 (-50 to +60 degC); least significant bit is 1 mm s-1 (0.025 degC embedded code version 4; 0.002 degC embedded code version 3)" | Sec 2.2, p. 3, PDF 13 | `[CITED]` |
| C30 | Binary speed of sound | 1 mm/s step about 340 m/s | "speed of sound (m s-1) = word 3 * 0.001 + 340.0" | Table B-6, p. B-5, PDF 63 | `[CITED]` |
| C31 | Diagnostic word, flags | bits 15-12 | "Bits 15 through 12 are four diagnostic flags" | Sec 10.4, p. 22, PDF 32 | `[CITED]` |
| C32 | Diagnostic word, counter | bits 5-0 | "bits 5 through 0 are a counter" | Sec 10.4, p. 22, PDF 32 | `[CITED]` |
| C33 | Counter range | 0 to 63 | "The minimum value is 0 and the maximum is 63." | App B, p. B-5, PDF 63 | `[CITED]` |
| C34 | Flag b15 | path speed-of-sound spread above 2.360 m/s | "Difference in the speed of sound between the three non-orthogonal axes is greater than 2.360 m s-1" | Table 10-6, p. 23, PDF 33 | `[CITED]` |
| C35 | Flag b14 | poor signal lock | "Poor signal lock" | Table 10-6, p. 23, PDF 33 | `[CITED]` |
| C36 | Flag b13 | amplitude too high | "Sonic signal amplitude too high" | Table 10-6, p. 23, PDF 33 | `[CITED]` |
| C37 | Flag b12 | amplitude too low | "Sonic signal amplitude too low" | Table 10-6, p. 23, PDF 33 | `[CITED]` |
| C38 | Fault: no response | NaN or -99999 | "NaN or -99999, Anemometer does not Respond." | Sec 10.4, p. 22, PDF 32 | `[CITED]` |
| C39 | Fault 61440 | lost trigger | "61440, Lost Trigger." | Sec 10.4, p. 22, PDF 32 | `[CITED]` |
| C40 | Lost-trigger onset | three seconds without trigger | "it will enter a lost trigger state after three seconds." | Sec 8, p. 16, PDF 26 | `[CITED]` |
| C41 | Fault 61503 | no new data | "61503, No Data Available. The anemometer has no new data." | Sec 10.4, p. 22, PDF 32 | `[CITED]` |
| C42 | Fault 61503, overrun | also reported on overrun | "The latter will cause the CSAT3 to output 61503 as the diagnostic word, indicating an overrun" | Sec 8, p. 16, PDF 26 | `[CITED]` |
| C43 | Fault 61441 | SDM communications error | "61441, SDM Communications Error." | Sec 10.4, p. 23, PDF 33 | `[CITED]` |
| C44 | Fault 61442 | wrong embedded code | "61442, Wrong CSAT3 embedded code." | Sec 10.4, p. 23, PDF 33 | `[CITED]` |
| C45 | Fault 61502 | example program's code for a NaN word | "If ( diag_csat = NaN ) Then ( diag_csat = 61502 )" | Example program, p. 27, PDF 37 | `[CITED]` |

The manual lists no signal-strength output `[DERIVED]` from a search
of `csat3.txt` (outputs: ux, uy, uz, c; Sec 2.1, p. 2, PDF 12
`[CITED]`); the head's signal quality is carried
only by flags b14-b12 (C35-C37) `[DERIVED]` from C31, C35-C37.

## LI-7500DS -- open-path CO2/H2O analyzer [@LICOR7500DS]

| ID | Quantity | Value | Quote | Locus | Label |
|---|---|---|---|---|---|
| L1 | Optical path length | 12.5 cm | "Path Length: 12.5 cm" | App B, p. B-2, PDF 212 | `[CITED]` |
| L2 | Separation, north | entered per site, sign north positive | "Northward Separation (cm): North (positive) or south (negative) distance between the anemometer and gas analyzer sample volumes." | Sec 4, p. 4-10, PDF 78 | `[CITED]` |
| L3 | Separation, east | entered per site, sign east positive | "Eastward Separation (cm): East (positive) or west (negative) distance between the anemometer and gas analyzer sample volumes." | Sec 4, p. 4-10, PDF 78 | `[CITED]` |
| L4 | Separation, vertical | midline to midline | "Vertical Separation (cm): Distance between the anemometer midline and the gas analyzer sample midline." | Sec 4, p. 4-11, PDF 79 | `[CITED]` |
| L5 | Fundamental sampling rate | 150 Hz | "Fundamental Gas Sampling Rate: 150 Hz" | App B, p. B-2, PDF 212 | `[CITED]` |
| L6 | Bandwidth options | 5, 10 or 20 Hz | "Bandwidth: 5, 10, or 20 Hz; software selectable" | App B, p. B-2, PDF 212 | `[CITED]` |
| L7 | Bandwidth, role | digital filter averaging | "Bandwidth (5, 10 or 20 Hz) determines the signal averaging done by the digital filter." | Sec 10, p. 10-13, PDF 171 | `[CITED]` |
| L8 | Bandwidth, definition | frequency where indicated amplitude is 0.707 of real | "Bandwidth is the frequency at which the indicated amplitude is 0.707 of the real amplitude" | Sec 10, p. 10-13, PDF 171 | `[CITED]` |
| L9 | Bandwidth against rate | half the sampling rate | "Thus, if you are sampling at 10 Hz, the bandwidth is 5 Hz." | Sec 10, p. 10-13, PDF 171 | `[CITED]` |
| L10 | Bandwidth against log rate | half the log rate | "verify that the Bandwidth is 1/2 of the log rate." | Sec 5, p. 5-2, PDF 82 | `[CITED]` |
| L11 | Typical log rate | 10 samples/s | "Log Rate (sps; samples per second): 10.0, typically." | Sec 5, p. 5-1, PDF 81 | `[CITED]` |
| L12 | Vibration sensitivity | 150 Hz +/- bandwidth | "The LI-7500DS is vibration sensitive to frequencies of 150 Hz +/- the bandwidth." | Sec 5, p. 5-20, PDF 100 | `[CITED]` |
| L13 | Output | Ethernet | "Outputs: Ethernet" | App B, p. B-2, PDF 212 | `[CITED]` |
| L14 | Operating temperature | -25 to 50 C | "Operating Temperature Range: -25 to 50 degC (-40 to 50 degC verification on request)" | App B, p. B-2, PDF 212 | `[CITED]` |
| L15 | CO2 calibration range | 0 to 3000 umol/mol | "Calibration Range: 0 to 3000 umol mol-1" | App B, p. B-1, PDF 211 | `[CITED]` |
| L16 | H2O calibration range | 0 to 60 mmol/mol | "Calibration Range: 0 to 60 mmol mol-1" | App B, p. B-1, PDF 211 | `[CITED]` |
| L17 | Diagnostic value, size | 1 byte, 0 to 255 | "The cell diagnostic value is a 1 byte unsigned integer (value between 0 and 255)" | Sec 8, p. 8-5, PDF 131 | `[CITED]` |
| L18 | Diagnostic bits 0-3 | signal strength | "Value x 6.67 = Signal Strength." | Sec 8, p. 8-5, PDF 131 | `[CITED]` |
| L19 | Diagnostic bit 6 | detector temperature near setpoint when set | "1 = Detector temperature is near setpoint" | Sec 8, p. 8-5, PDF 131 | `[CITED]` |
| L20 | Diagnostic bit 7 | chopper temperature near setpoint when set | "1 = Chopper wheel temperature is near setpoint" | Sec 8, p. 8-5, PDF 131 | `[CITED]` |
| L21 | Diagnostic example | 125 means chopper not OK | "A value of 125 (01111101) indicates Chopper not ok" | Sec 8, p. 8-6, PDF 132 | `[CITED]` |
| L22 | Diagnostic bit sense | 1 = OK, 0 = bad | "Bit (0=BAD, 1=OK)" | Table 8-1, p. 8-8, PDF 134 | `[CITED]` |
| L23 | Diagnostic value, logged file | 0 to 8191 | "Diagnostic value 0-8191" | Table 6-2, p. 6-3, PDF 103 | `[CITED]` |
| L24 | Signal strength outputs | CO2 (Sc) and H2O (Sw) | "CO2 Signal Strength (Sc) [...] H2O Signal Strength (Sw)" | Table 6-2, p. 6-4, PDF 104 | `[CITED]` |
| L25 | Signal strength scale | 0 to 100, 100 clean | "Sc is on a 0 to 100 scale, where 100 is clean." | Sec 8, p. 8-6, PDF 132 | `[CITED]` |
| L26 | Signal strength threshold | none absolute; upper 90s advised | "There is no absolute guideline for good or bad signal strength, but 100% indicates very clean optics [...] Typically, signal strength values should be maintained in the upper 90's for consistent performance." | Sec 5, p. 5-17, PDF 97 | `[CITED]` |

The bit map on p. 8-5 assigns bits 0-3 to signal strength, 4 to Sync,
5 to PLL, 6 to Detector and 7 to Chopper (L17-L20) `[CITED]` p. 8-5,
PDF 131. The manual states no filter or pipeline delay for the
LI-7500DS in any section: the word "delay" occurs only in GPS and
zero-calibration text `[DERIVED]` from a search of the whole
extraction (`7500DS_Manual_16770.txt`).

## IRGASON with EC100 -- integrated sonic and open-path analyzer [@CampbellIRGASON]

| ID | Quantity | Value | Quote | Locus | Label |
|---|---|---|---|---|---|
| I1 | Optical path length | 15.37 cm | "Optical path length: 15.37 cm (6.05 in)" | Sec 5.3, p. 5, PDF 16 | `[CITED]` |
| I2 | Sonic path length | 10.0 cm | "Sonic path length: 10.0 cm (3.9 in)" | Sec 5.3, p. 5, PDF 16 | `[CITED]` |
| I3 | Sonic path angle | 30 deg from vertical | "Sonic path angle from vertical: 30deg" | Sec 5.3, p. 5, PDF 16 | `[CITED]` |
| I4 | Gas-sonic separation | 0 cm | "Spatial separation between gas analyzer and sonic volumes: 0 cm" | Sec 5.3, p. 5, PDF 16 | `[CITED]` |
| I5 | Measurement rate | 60 Hz | "Measurement Rate: 60 Hz" | Sec 5.1, p. 3, PDF 14 | `[CITED]` |
| I6 | Output bandwidth, spec | 5, 10, 12.5 or 20 Hz | "Output bandwidth1: 5, 10, 12.5, or 20 Hz" | Sec 5.1, p. 3, PDF 14 | `[CITED]` |
| I7 | Bandwidth, settings | 5, 10, 12.5, 20 or 25 Hz | "The EC100 has a user-selectable low-pass filter to select the bandwidth (5, 10, 12.5, 20, or 25 Hz)." | Sec 7.2, p. 16, PDF 27 | `[CITED]` |
| I8 | Bandwidth, appendix | 5, 10, 12.5, 20 and 25 Hz | "The available filter bandwidths are 5, 10, 12.5, 20, and 25 Hz." | App A, p. A-1, PDF 58 | `[CITED]` |
| I9 | Factory bandwidth | 20 Hz | "the factory default for the EC100 bandwidth (20 Hz)" | App A, p. A-1, PDF 58 | `[CITED]` |
| I10 | Bandwidth, factory setting table | 20 Hz | "Bandwidth 20 Hz" | Table 7-1, p. 15, PDF 26 | `[CITED]` |
| I11 | Bandwidth rule | in general below half the sample rate; when the filter is matched to the scan rate, half the logger scan rate | "bandwidth must be less than half of the sample rate (datalogger scan rate) [...] In this case, the selected bandwidth should be one-half of the sample rate (datalogger scan rate)" | App A, p. A-1, PDF 58 | `[CITED]` |
| I12 | Bandwidth for spectra | Nyquist of the logger rate or of the unprompted output rate | "the bandwidth should be set to the Nyquist frequency, which is half the datalogger sample rate (for SDM output) or half the unprompted output rate (for USB and RS-485 output)." | Sec 7.2, p. 16, PDF 27 | `[CITED]` |
| I13 | Logger-prompted rate | 1 to 60 Hz | "Datalogger-prompted output rate: 1 to 60 Hz" | Sec 5.1, p. 3, PDF 14 | `[CITED]` |
| I14 | SDM sample rate | inverse of the scan interval | "The sample rate for SDM output is determined by the inverse of the datalogger scan interval" | Sec 8.1, p. 22, PDF 33 | `[CITED]` |
| I15 | SDM down-sampling | 60 Hz to the scan rate | "down-sampled to the datalogger scan rate through SDM communications" | App A, p. A-2, PDF 59 | `[CITED]` |
| I16 | Unprompted rate, spec | 10, 25 or 50 Hz | "Output rate1: 10, 25 or 50 Hz" | Sec 5.1, p. 3, PDF 14 | `[CITED]` |
| I17 | Unprompted rate, appendix | 10, 25 or 50 Hz | "to a user-selectable rate of 10, 25, or 50 Hz" | App A, p. A-2, PDF 59 | `[CITED]` |
| I18 | Unprompted rate, RS-485 | 10, 20 or 60 Hz | "Output rate: 10, 20 or 60 Hz7" | Sec 5.2, p. 5, PDF 16 | `[CITED]` |
| I19 | Unprompted rate, settings | 10, 20 or 60 Hz | "but it may be set to 10, 20, or 60 Hz." | Sec 7.5, p. 17, PDF 28 | `[CITED]` |
| I20 | Gas and sonic sync | synchronous in the EC100 | "The EC100 electronics synchronously sample the gas analyzer and sonic anemometer of the IRGASON." | App A, p. A-1, PDF 58 | `[CITED]` |
| I21 | Filter delay, form | constant across the pass band | "The EC100 provides a constant time delay for all spectral components within each filter's pass band." | App A, p. A-1, PDF 58 | `[CITED]` |
| I22 | Filter delay by bandwidth | 5 Hz 800 ms; 10 Hz 400 ms; 12.5 Hz 317 ms; 20 Hz 200 ms | "Bandwidth (Hz) Time Delay (ms) 5 800 10 400 12.5 317 20 200" | Table A-1, p. A-3, PDF 60 | `[CITED]` |
| I23 | Delay in scans, 25 Hz scan | five scans at 20 Hz bandwidth | "delay the non-EC100 data by five datalogger scans" | App A, p. A-1, PDF 58 | `[CITED]` |
| I24 | Delay in scans, 10 Hz scan | two scans at 20 Hz bandwidth | "delay the non-EC100 data by two datalogger scans to match the EC100 data" | App A, p. A-1, PDF 58 | `[CITED]` |
| I25 | Ts equation constants | gamma_d = 1.4, R_d = 287.04 J K-1 kg-1 | "The sonic virtual temperature in degrees Celsius, is given by equation (9), where gammad = 1.4 and Rd = 287.04 J.K-1.kg-1." | Sec 11.1.2, p. 44, PDF 55 | `[CITED]` |
| I26 | Crosswind correction | online, text names the CSAT3 | "The CSAT3 corrects online for the effect of wind blowing perpendicular to the sonic path." | Sec 11.1.2, p. 42, PDF 53 | `[CITED]` |
| I27 | Operating temperature | -30 to 50 C | "Operating temperature: -30 to 50 degC" | Sec 5.1, p. 3, PDF 14 | `[CITED]` |
| I28 | Wind output range | +/-65.553 m/s | "Full scale wind: +/-65.553 m/s" | Sec 5.1, p. 4, PDF 15 | `[CITED]` |
| I29 | Ts reporting range | -50 to 60 C | "Sonic temperature: -50 to 60 degC" | Sec 5.1, p. 4, PDF 15 | `[CITED]` |
| I30 | Ts precision | 0.025 C rms | "Sonic measurement precision4 [...] Sonic temperature: 0.025 degC [...] 4 noise rms" | Sec 5.1, p. 4, PDF 15 | `[CITED]` |
| I31 | CO2 calibrated range | 0 to 1000 umol/mol | "0 to 1000 umol.mol-1" | Sec 5.1, p. 3, PDF 14 | `[CITED]` |
| I32 | Pressure range | 70 to 106 kPa | "70 to 106 kPa" | Sec 5.1, p. 3, PDF 14 | `[CITED]` |
| I33 | Signal strength range | nominally 0 to 1 | "Nominally 0.0 to 1.0" | Table 8-1, p. 23, PDF 34 | `[CITED]` |
| I34 | Signal strength, cleaning | below 0.7 or 70 % of original | "The windows of the analyzer should be cleaned if the signal strength for CO2 or H2O drops below 0.7 or 70% of the original value." | Sec 9.3, p. 28, PDF 39 | `[CITED]` |
| I35 | Diagnostic words | bit set while the condition lasts | "When a certain condition is detected, the corresponding bit is set. The value remains set until the event that caused the condition is no longer present." | Sec 10.2, p. 36, PDF 47 | `[CITED]` |
| I36 | Sonic bit 0 (1) | amplitude too low | "0 0x1 1 Low Amp Amplitude is too low" | Table 10-2, p. 36, PDF 47 | `[CITED]` |
| I37 | Sonic bit 1 (2) | amplitude too high | "1 0x2 2 High Amp Amplitude is too high" | Table 10-2, p. 36, PDF 47 | `[CITED]` |
| I38 | Sonic bit 2 (4) | poor signal lock | "2 0x4 4 Tracking Poor signal lock" | Table 10-2, p. 36, PDF 47 | `[CITED]` |
| I39 | Sonic bit 3 (8) | delta temperature out of limits | "3 0x8 8 Hi 3 Axis DC Delta temperature exceeds limits" | Table 10-2, p. 36, PDF 47 | `[CITED]` |
| I40 | Sonic bit 4 (16) | acquiring signals | "4 0x10 16 Acquiring Acquiring ultrasonic signals" | Table 10-2, p. 36, PDF 47 | `[CITED]` |
| I41 | Sonic bit 5 (32) | head calibration signature error | "5 0x20 32 Cal Mem Err Sonic head calibration signature error" | Table 10-2, p. 36, PDF 47 | `[CITED]` |
| I42 | Gas bit 0 (1) | some flag active | "Data are suspect (there is an active diagnostic flag)" | Table 10-3, p. 37, PDF 48 | `[CITED]` |
| I43 | Gas bit 18 (262144) | CO2 signal low | "18 0x40000 262144 CO2 Io Ratio CO2 signal level too low" | Table 10-3, p. 37, PDF 48 | `[CITED]` |
| I44 | Gas bit 19 (524288) | H2O signal low | "19 0x80000 524288 H2O Io Ratio H2O signal level too low" | Table 10-3, p. 37, PDF 48 | `[CITED]` |
| I45 | SDM error value | -99999 on signature error | "A value of -99999 will be loaded into Dest(1) if a signature error on SDM data occurs." | Sec 10.2, p. 35, PDF 46 | `[CITED]` |

## Ambiguities in the sources

Recorded, not resolved.

- IRGASON unprompted output rate: "10, 25 or 50 Hz" (I16, I17) against
  "10, 20 or 60 Hz" (I18, I19) `[CITED]` I16-I19.
- IRGASON bandwidths: four in the specification (I6, and the
  EC100Configure parameter list on p. 39, PDF 50, "Bandwidth: 5 = 5 Hz,
  10 = 10 Hz, 12 = 12.5 Hz, 20 = 20 Hz") against five in Sec 7.2 and
  App A (I7, I8) `[CITED]` I6-I8, p. 39, PDF 50. Table A-1 (I22) has no
  25 Hz row, so the delay of a 25 Hz setting is not stated `[DERIVED]`
  from I7, I8, I22.
- IRGASON optical path: 15.37 cm in Sec 5.3 (I1) against "propagating
  along a 15 cm optical path" in Sec 11.2, p. 44, PDF 55 `[CITED]` I1,
  p. 44, PDF 55.
- Full-scale wind: +/-65.553 m/s for the IRGASON (I28) against
  +/-65.535 m/s for the CSAT3 (C28) `[CITED]` I28, C28; the manuals do
  not say whether the heads differ or one is a transposition
  `[DERIVED]` from C28, I28.
- IRGASON Sec 11.1 names the CSAT3 for the crosswind correction (I26)
  and for the transformation matrix ("unique for each CSAT3", p. 42,
  PDF 53) `[CITED]` I26, p. 42, PDF 53. That the IRGASON sonic applies
  the same online correction is read from this text, not stated for
  the IRGASON by name `[DERIVED]` from I26.
- IRGASON App A, p. A-2, PDF 59: SDM sync "to within +/-5 ms (plus or
  minus one-half of the inverse of 60 Hz)" `[CITED]` p. A-2, PDF 59;
  one-half of 1/60 s is 8.3 ms `[DERIVED]` from that quote, so the
  number and its gloss disagree.
- CSAT3 speed-of-sound LSB: 0.025 C for 1 mm/s in Sec 2.2 (C29)
  against 0.0017 C at 25 C in Table B-6, p. B-5, PDF 63 `[CITED]` C29,
  Table B-6, p. B-5, PDF 63. With c = 346 m/s at 298 K,
  dc/dT = c/(2T) = 0.58 m s-1 K-1, so 1 mm/s is 0.0017 C `[DERIVED]`
  from C29, C30; 0.025 C is the code-version-4 resolution (C25),
  15 mm/s `[CITED]` C25.
- CSAT3 fault 61503: "No Data Available" in Sec 10.4 (C41) and an
  overrun in Sec 8 (C42) `[CITED]` C41, C42. Code 61502 appears only in
  the example program (C45), as the value the program assigns to a NaN
  word `[CITED]` C45.
- LI-7500DS diagnostic range: 0-255 for the cell diagnostic byte
  (L17) against 0-8191 in the logged-variable table (L23) `[CITED]`
  L17, L23. The manual does not say what the upper bits of the logged
  value carry `[DERIVED]` from L17, L23.
- LI-7500DS signal strength: 0-100 % (L25, L26) where the IRGASON
  reports 0-1 (I33) `[CITED]` L25, L26, I33; the two scales differ by a
  factor of 100 `[DERIVED]` from L25, I33.

## Read against the code

Verdicts: agrees, differs, not represented.

| Spec rows | Code | Verdict |
|---|---|---|
| C13-C17 CSAT3 two-scan delay; I21-I24 EC100 filter delay | [utespac/flux/engine.py](../../utespac/flux/engine.py):57 `scan_freq`, :61 `lag_search = "off"`, :62 `lag_max_s = 2.0`, :120 `_align`; [utespac/flux/covariance.py](../../utespac/flux/covariance.py):151 `lag_max_abs_cov` | not represented. No fixed instrument delay is applied; a covariance-peak search, off by default, can absorb either delay within +/-2 s when enabled. |
| C19-C23 CSAT3 Ts crosswind-corrected in the head | [sonic_temperature_flux.md](sonic_temperature_flux.md):39-41, :54 | agrees |
| I26 IRGASON crosswind | [sonic_temperature_flux.md](sonic_temperature_flux.md):39-41, :54 | agrees with the text, which names the CSAT3 (see Ambiguities) |
| C29, I29 Ts range -50 to 60 C; C27, I27 operating -30 to 50, cold-shifted -40 to 40 C | [utespac/config/qc.toml](../../utespac/config/qc.toml):44 `Tson = [-20, 80]` | differs. The upper bound exceeds the reporting range; the lower bound sits above the operating floor. The bound is `[ASSUMED]` from UTESpac v5.0. |
| C28, I28 wind range +/-65.5 m/s | [utespac/config/qc.toml](../../utespac/config/qc.toml):41-43 `u`, `v` [-50, 50], `w` [-10, 10] | not represented (plausibility bounds inside the output range) |
| I34 IRGASON signal floor 0.7 | [utespac/config/qc.toml](../../utespac/config/qc.toml):68-69 `H2OminSignal`, `CO2minSignal` = 0.7; [utespac/flux/levels.py](../../utespac/flux/levels.py):274-275, :301-302 `np.less_equal` | agrees on the value; the code also flags equality. The manual gives "drops below 0.7" as a window-cleaning criterion (I34); its alternative, 70 % of the original value, is not represented. |
| I35-I44 IRGASON gas diagnostic word | [utespac/config/qc.toml](../../utespac/config/qc.toml):70 `meanGasDiagnosticLimit = 0.1`; [utespac/flux/levels.py](../../utespac/flux/levels.py):277, :304 | not represented as a manual value. Any set bit makes the word at least 1; a period-mean limit of 0.1 is not stated in the manual. |
| C31-C37 CSAT3 flags; I36-I41 IRGASON sonic bits | [utespac/config/qc.toml](../../utespac/config/qc.toml):71 `meanSonicDiagnosticLimit = 50`; [utespac/flux/levels.py](../../utespac/flux/levels.py):127-134; [utespac/find_global_pf.py](../../utespac/find_global_pf.py):290 | not represented. The limit is not in either manual, and its meaning depends on how the logger stores the word (CSAT3 flags sit in bits 15-12 with a 0-63 counter below; IRGASON bits 0-5). |
| L17-L22 LI-7500DS diagnostic byte, 255 all OK | [utespac/config/qc.toml](../../utespac/config/qc.toml):72 `meanLiGasDiagnosticLimit = 220`; [utespac/flux/levels.py](../../utespac/flux/levels.py):281-286, :306-310 | agrees on the code comment's "full strength is 255" (levels.py:281), which is `[DERIVED]` from L18-L20 and the p. 8-5 bit map (bits 4, 5), sense L22, not stated in the manual; the limit 220 is not stated. With bits 4-7 all OK the byte is at least 240 `[DERIVED]`. |
| C38-C44 CSAT3 fault codes | [utespac/raw_processing/toa5_tower.py](../../utespac/raw_processing/toa5_tower.py):206-208 `SONIC_FAULT_CODES = (61440, 61441, 61503)`, :211 `SONIC_SENTINEL = -99999.0`, :281-285 | differs in wording: the comment reads 61503 as "the head did not respond", which the manual gives as no new data or overrun (C41, C42), and gives "NaN or -99999" for no response (C38). 61442 and 61502 are not represented; flag bits b15-b12 are not masked per sample. |
| C1, I1, I2, L1 path lengths; I4, L2-L4 separations | [utespac/config/flux.toml](../../utespac/config/flux.toml):22-25 `sonicPathLength`, `irgaPathLength`, `irgaSeparationLateral`, `irgaSeparationLongitudinal`; [utespac/stages.py](../../utespac/stages.py):380-383; [utespac/flux/engine.py](../../utespac/flux/engine.py) `_massman_factors` | not represented as values: flux config keys with no defaults, required under `spectralCorrection = "massman2000"`; `stages.flux` reads them into `FluxOptions` and `_massman_factors` passes them to `massman2000_factor`. One geometry serves every sonic level. The separations are lateral and longitudinal to the wind, not the LI-COR north/east frame of L2-L3; see [spectral_correction.md](spectral_correction.md). |
| L5-L10 LI-7500DS bandwidth; I6-I12 EC100 bandwidth | none | not represented |

## Proposed siteInfo fields

A proposal, not code. Per-sonic fields in the `[[sonics]]` block of
`siteInfo.toml` (the synthetic fixture's form), snake case beside the
existing `height`, `orientation`, `manufacturer` `[HOUSE]`.

| Field | Unit | Supporting rows |
|---|---|---|
| `model` (CSAT3, IRGASON) | -- | section headings; overlaps the existing `manufacturer` (1 = CSAT/IRGASON, [utespac/site_config.py](../../utespac/site_config.py):43), which does not tell the two heads apart |
| `embedded_code_version` (CSAT3) | -- | C25, C26, C29 |
| `execution_parameter` (CSAT3 sampling mode) | -- | C3, C8, C11, C12, C17 |
| `pipeline_delay_scans` | scans | C13-C16 |
| `path_length_m` | m | I2 (the CSAT3 manual gives the vertical extent, C1) |
| `irga_model` (LI-7500DS, IRGASON) | -- | section headings |
| `irga_path_length_m` | m | L1, I1 |
| `irga_bandwidth_hz` | Hz | L6-L10, I6-I12 |
| `irga_filter_delay_ms` | ms | I21-I24 (not stated for the LI-7500DS) |
| `separation_north_m`, `separation_east_m`, `separation_vertical_m` | m | L2-L4, I4 |
| `irga_signal_floor` | instrument scale (0-1 or 0-100) | I33, I34, L25, L26; overlaps `H2OminSignal`, `CO2minSignal` in [utespac/config/qc.toml](../../utespac/config/qc.toml):68-69 |
