"""Readable driver for the supplied CarbonMetabolism equations (GPL-3.0-or-later).

Concentrations: mM. Time: s. Raw rates: mM/s. Area rates: raw * 30,
following CM_OUT. CO2 input is intercellular Ci in ppm, not ambient Ca.
"""
from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_ivp
from .equations import Equations
from .matlab import M, zeros

STATE_NAMES = ('RuBP PGA DPGA T3P FBP E4P S7P SBP ATP NADPH CO2 O2 HexP PenP '
               'GCEA GCA PGCA GCAc GOAc SERc GLYc HPRc GCEAc '
               'T3Pc FBPc HexPc F26BPc ATPc ADPc OPOPc UDPGc UTPc SUCP SUC PGAc').split()
FLUX_NAMES = ('v1 v2 v3 reserved4 v5 v6 v7 v8 v9 v10 v13 v16 v23 v31 v32 v33 '
              'v111 v112 v113 v121 v122 v123 v124 v131 v1in v2out '
              'v51 v52 v55 v56 v57 v58 v59 v60 reserved35 v62 '
              'vdhap_in vgap_in vpga_in vpga_use vatpf').split()
ENZYMES = {'Rubisco': ('V1',), 'PGK': ('V2',), 'GAPDH': ('V3',),
           'FBA': ('V5', 'V8'), 'FBPase': ('V6',), 'TK': ('V7', 'V10'),
           'SBPase': ('V9',), 'PRK': ('V13',), 'AGPase': ('V23',),
           'GDC': ('V131',), 'GOX': ('V121',), 'SGAT': ('V122',),
           'HPR': ('V123',), 'GGAT': ('V124',), 'SPS': ('V56',)}


class C3Model:
    """Generic C3 physiological teaching revision of the 35-state network.

    Default physiology shares bound RuBP between carboxylation/oxygenation,
    includes lumped reassimilation ATP cost and external-reserve respiration.
    formulation='original' retains the unchanged source kinetics/output.

    enzyme_factors scales Vmax at fixed kcat (thus proportional enzyme amount).
    There is no fitted protein abundance, kcat table, or fixed nitrogen budget.
    Source Vcmax25 instead scales multiple enzymes AND the phosphate pool.
    """
    def __init__(self, light=1000., ci=280., temperature=25., vcmax25=80.,
                 jmax25=160., *, enzyme_factors=None, sbpase_rate=None, fbpase_rate=None,
                 formulation='physiology', oxygen=.21, physiology_parameters=None):
        inputs = np.array([light, ci, temperature, vcmax25, jmax25], dtype=float)
        if not np.all(np.isfinite(inputs)) or light < 0 or ci <= 0 or min(vcmax25, jmax25) <= 0:
            raise ValueError('Light must be nonnegative; Ci, Vcmax25 and Jmax25 must be positive. All inputs must be finite.')
        if not 5 <= temperature <= 45:
            raise ValueError('Temperature must be within 5–45 °C. Physiological checks cover 15–35 °C only.')
        if formulation not in ('original','physiology'):
            raise ValueError('formulation must be original or physiology.')
        if not np.isfinite(oxygen) or not 0<=oxygen<=1:
            raise ValueError('oxygen must be a volume fraction between 0 and 1.')
        if formulation=='original' and (oxygen!=.21 or physiology_parameters is not None):
            raise ValueError('Original mode retains 21% oxygen and the original parameters. Use physiology mode for these options.')
        self.light = float(light); self.ci = float(ci); self.temperature = float(temperature)
        self.formulation=formulation
        self.respiration=0.
        if formulation=='physiology':
            from .physiology_equations import PhysiologyEquations
            from .physiology import PhysiologyParameters
            e=PhysiologyEquations()
            e.physiology=physiology_parameters or PhysiologyParameters()
            e.oxygen_fraction=float(oxygen)
            self.respiration=e.physiology.respiration(self.temperature)
        else:e=Equations()
        self.equations = e
        e.CO2_Env = self.ci; e.CO2_cond = self.ci / 30000
        e.LI = self.light / 30; e.Tp = self.temperature
        e.Jmax = float(jmax25) / 30
        e.alfa = .85; e.fc = .15; e.Theta = .7
        e.beta = .75 if formulation=='physiology' else .7519
        e.GP = 0; e.GRNC = 0; e.GRNT = 0
        self.initial = e.CM_Ini(1, float(vcmax25)).vector()
        e.IModelCom()
        e.PR_PS_com = 1; e.ATPActive = 0; e.PSPR_SUCS_com = 1
        self.sucs_param = M([1., 1.])
        # Original logs never feed the equations; keep a single latest entry.
        for prefix in ['PS', 'PR', 'SUCS']:
            setattr(e, prefix + '_TIME_N', 1)
            setattr(e, prefix + '_OLD_TIME', np.inf)
            setattr(e, prefix + '_VEL', zeros(1, 1))
        e.SUCS_CON = zeros(1, 2)
        for enzyme, factor in (enzyme_factors or {}).items():
            if enzyme not in ENZYMES or not np.isfinite(factor) or factor <= 0:
                raise ValueError('Unknown enzyme or nonpositive enzyme multiplier: ' + enzyme)
            for name in ENZYMES[enzyme]:
                setattr(e, name, getattr(e, name) * factor)
        if sbpase_rate is not None:
            if not callable(sbpase_rate):
                raise TypeError('sbpase_rate must be callable.')
            e.student_sbpase = sbpase_rate
        if fbpase_rate is not None:
            if not callable(fbpase_rate):
                raise TypeError('fbpase_rate must be callable.')
            e.student_fbpase = fbpase_rate

    def set_light(self, par):
        par = float(par)
        if not np.isfinite(par) or par < 0:
            raise ValueError('PAR must be finite and nonnegative.')
        self.equations.LI = par / 30

    def rhs(self, t, y):
        return self.equations.CM_mb(float(t), M(y), 0, self.sucs_param).vector()

    def fluxes(self, t, y, *, area=True):
        self.equations.cdn(float(t))
        v = self.equations.CM_Rate(float(t), M(y), self.sucs_param).vector()
        return v * (30 if area else 1)

    def assimilation(self, t, y):
        v = self.fluxes(t, y)
        # Biochemical flux difference follows CM_OUT. The physiology mode
        # subtracts external-reserve Rd once; original mode retains Rd=0.
        return float(v[0] - v[23] - self.respiration)


@dataclass
class Result:
    time: np.ndarray
    states: np.ndarray
    fluxes: np.ndarray
    light: np.ndarray
    nfev: int
    respiration: float = 0.
    formulation: str = 'original'

    @property
    def assimilation(self):
        return self.biochemical_assimilation - self.respiration

    @property
    def biochemical_assimilation(self):
        return self.fluxes[:, 0] - self.fluxes[:, 23]

    def concentration(self, name):
        return self.states[:, STATE_NAMES.index(name)]

    def flux(self, name):
        return self.fluxes[:, FLUX_NAMES.index(name)]


def simulate(model, duration=3000., *, times=None, initial=None,
             light_program=None, breakpoints=(), rtol=1e-7, atol=1e-10):
    """BDF integration; split at each discontinuous light change explicitly.

    No pool clipping or artificial stabilization. For classroom comparisons,
    warm an unmodified model once, then pass its final states as initial.
    """
    duration = float(duration)
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError('duration must be finite and positive.')
    times = np.linspace(0, duration, 151) if times is None else np.asarray(times, dtype=float)
    if times.ndim != 1 or not len(times) or not np.all(np.isfinite(times)) or times[0] < 0 or times[-1] > duration or np.any(np.diff(times) <= 0):
        raise ValueError('Sampling times must be finite, strictly increasing and within the simulation interval.')
    y = model.initial.copy() if initial is None else np.asarray(initial, dtype=float).copy()
    if y.shape != (35,) or not np.all(np.isfinite(y)):
        raise ValueError('The initial state must contain 35 finite values.')
    breaks = np.asarray(list(breakpoints), dtype=float)
    if not np.all(np.isfinite(breaks)):
        raise ValueError('Light-switching times must be finite.')
    points = [0.] + sorted(set(float(x) for x in breaks if 0 < x < duration)) + [duration]
    states = np.empty((len(times), 35)); evaluations = 0
    program = light_program or (lambda t: model.light)
    for start, stop in zip(points[:-1], points[1:]):
        def rhs(t, state):
            # At a discontinuity the previous segment uses its left limit.
            sample = np.nextafter(stop, start) if t >= stop else t
            model.set_light(program(sample))
            return model.rhs(t, state)
        solution = solve_ivp(rhs, (start, stop), y, method='BDF',
                             rtol=rtol, atol=atol, dense_output=True)
        if not solution.success:
            raise RuntimeError('Integration failed: ' + solution.message)
        mask = (times >= start) & (times <= stop)
        states[mask] = solution.sol(times[mask]).T
        y = solution.y[:, -1]; evaluations += solution.nfev
    par = np.array([program(float(t)) for t in times], dtype=float)
    fluxes = []
    for t, state, light in zip(times, states, par):
        model.set_light(light)
        fluxes.append(model.fluxes(t, state))
    return Result(times, states, np.asarray(fluxes), par, evaluations,
                  model.respiration,model.formulation)
