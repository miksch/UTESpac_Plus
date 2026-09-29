"""The flux stage as separate, testable pieces (migration step 4).

``utespac.stages.flux`` orchestrates them on a :class:`~utespac.model.Run`:
:mod:`.reference` (site-reference pressure, temperature, humidity and
density per period), :mod:`.levels` (what one sonic level brings to the
computation), :mod:`.engine` (the per-period covariances, WPL terms and
quality flags) and :mod:`.tables` (the named output columns and their
legacy matrix/header form).

Three modules hold array-level kernels that do not depend on the run
model: :mod:`.covariance` (block detrending, lagged covariances, despiking
and instrument-noise variance), :mod:`.quality` (stationarity, the Foken
classes, random error and signal-strength screening) and :mod:`.density`
(air state and the WPL terms of an open-path scalar flux).
"""
