"""Small, readable experiment helpers for the generic C3 teaching model.

No model equations are changed. Static treatments use each environment's native
initialization and the same simulation duration. Paired sunflecks share a
preconditioned state within each Ci/temperature combination. No results are
written to disk. Restart the kernel after editing model or helper source.
"""
from copy import deepcopy
import numpy as np
from .model import C3Model, simulate
from .physiology import carbon_residual

_STATIC = {}
_WARM = {}
_DYNAMIC = {}
_SOLVES = 0


def clear_experiment_cache():
    """Discard in-memory results after changing source or experiment assumptions."""
    global _SOLVES
    _STATIC.clear()
    _WARM.clear()
    _DYNAMIC.clear()
    _SOLVES = 0


def experiment_cache_info():
    return {'static_cases': len(_STATIC), 'preconditions': len(_WARM),
            'sunflecks': len(_DYNAMIC), 'model_integrations': _SOLVES}


def _solve(model, *args, **kwargs):
    global _SOLVES
    result = simulate(model, *args, **kwargs)
    _SOLVES += 1
    if not (np.isfinite(result.states).all() and np.isfinite(result.assimilation).all()):
        raise RuntimeError('The model returned a nonfinite trajectory.')
    return result


def static_environment(*, light=1000., ci=280., temperature=25., duration=3000.):
    """Return constant-environment endpoint metrics; not a claim of full steady state."""
    key = tuple(float(x) for x in (light, ci, temperature, duration))
    if key not in _STATIC:
        light, ci, temperature, duration = key
        model = C3Model(light=light, ci=ci, temperature=temperature)
        run = _solve(model, duration)
        vc = float(run.flux('v1')[-1])
        photorespiration = float(run.flux('v131')[-1])
        rd = float(run.respiration)
        net_a = float(run.assimilation[-1])
        np.testing.assert_allclose(net_a, vc-photorespiration-rd, rtol=1e-12, atol=1e-12)
        residual = float(carbon_residual(model, duration, run.states[-1]))
        if abs(residual) > 1e-8:
            raise RuntimeError('Carbon-boundary consistency check failed.')
        _STATIC[key] = {'PAR': light, 'Ci': ci, 'T': temperature, 'A': net_a,
                        'Vc': vc, 'PR': photorespiration, 'Rd': rd,
                        'time': duration, 'carbon_residual': residual}
    return deepcopy(_STATIC[key])


def sunfleck_environment(*, ci=280., temperature=25., low=300., high=1500.,
                         start=200., end=400., duration=600., settling=3000.):
    """Return dynamic outputs, with Ci and temperature constant within this run.

    Photon dose is integrated analytically. CO2 uptake is the trapezoidal
    integral of net assimilation on a one-second output grid, divided by 1000.
    Every environmental and timing parameter is part of the cache key.
    """
    key = tuple(float(x) for x in (ci, temperature, low, high, start, end, duration, settling))
    if key not in _DYNAMIC:
        ci, temperature, low, high, start, end, duration, settling = key
        if not (0 < start < end < duration and 0 <= low <= high):
            raise ValueError('Use 0 < start < end < duration and 0 <= low <= high.')
        warm_key = (ci, temperature, low, settling)
        if warm_key not in _WARM:
            precondition = C3Model(light=low, ci=ci, temperature=temperature)
            _WARM[warm_key] = _solve(precondition, settling).states[-1].copy()
        model = C3Model(light=low, ci=ci, temperature=temperature)
        times = np.unique(np.r_[np.arange(0., duration, 1.), duration, start, end])
        def light_program(t):
            return high if start <= t < end else low
        run = _solve(model, duration, initial=_WARM[warm_key], times=times,
                     light_program=light_program, breakpoints=[start, end])
        np.testing.assert_allclose(run.assimilation,
            run.flux('v1')-run.flux('v131')-run.respiration, rtol=1e-12, atol=1e-12)
        residual = float(carbon_residual(model, duration, run.states[-1]))
        if abs(residual) > 1e-8:
            raise RuntimeError('Carbon-boundary consistency check failed.')
        _DYNAMIC[key] = {
            'Ci': ci, 'T': temperature, 'photon_dose': low*duration+(high-low)*(end-start),
            'peak_A': float(run.assimilation.max()),
            'C': float(np.trapezoid(run.assimilation, run.time)/1000),
            'A_end': float(run.assimilation[-1]), 'time': run.time.copy(),
            'A_series': run.assimilation.copy(), 'PAR_series': run.light.copy(),
            'carbon_residual': residual}
    return deepcopy(_DYNAMIC[key])


def print_static_table(rows):
    """Rates are µmol CO2 m^-2 s^-1; Ci is ppm, T is °C, PAR is µmol photons m^-2 s^-1."""
    print('Case             PAR     Ci      T       Net A         Vc         PR         Rd')
    for label, row in rows:
        print(f'{label:<14} {row["PAR"]:6.0f} {row["Ci"]:6.0f} {row["T"]:6.1f}'
              f' {row["A"]:11.5f} {row["Vc"]:10.5f} {row["PR"]:10.5f} {row["Rd"]:10.5f}')


def print_dynamic_table(rows):
    """Cumulative C is mmol CO2 m^-2; photon dose is µmol photons m^-2."""
    print('Case                Ci      T   Photon dose      Peak A    Cumulative C')
    for label, row in rows:
        print(f'{label:<17} {row["Ci"]:6.0f} {row["T"]:6.1f} {row["photon_dose"]:13.0f}'
              f' {row["peak_A"]:11.5f} {row["C"]:15.5f}')
