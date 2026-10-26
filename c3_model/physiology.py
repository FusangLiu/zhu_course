"""Mechanistic Rubisco substrate sharing for the generic C3 teaching model.

GPL-3.0-or-later. This is a declared revision, not an unchanged MATLAB port.
References: Zhu et al. (2007), doi:10.1104/pp.107.103713, eqs.1.1.7–1.1.9;
von Caemmerer (2013), doi:10.1111/pce.12098, shared carboxylation/oxygenation.
The rest of the source metabolic network is retained, including fixed NADPH.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class PhysiologyParameters:
    # The original numeric Vmax/2 threshold implicitly used a 2 s^-1 turnover.
    # Expose it with units; this generic value is not a species measurement.
    kcat25: float = 2.0  # s^-1
    rd25: float = 1.0    # µmol CO2 m^-2 s^-1, external respiratory reserve
    rd_q10: float = 2.0
    km_atp_reassimilation: float = 0.1  # mM; generic teaching assumption

    def __post_init__(self):
        if not all(np.isfinite(x) for x in [self.kcat25,self.rd25,self.rd_q10,self.km_atp_reassimilation]) or self.kcat25<=0 or self.rd25<0 or self.rd_q10<=0 or self.km_atp_reassimilation<=0:
            raise ValueError('kcat25, rd_q10 and the ATP Km must be positive; rd25 must be nonnegative. All parameters must be finite.')

    def rubisco_pair(self, vmax_c, rubp_total, co2, oxygen, kc, ko, kr, regulation, temperature):
        """One enzyme–RuBP complex supplies both catalytic branches.

        For E+R <-> ER, solve E_total=E+ER and R_total=R+ER:
        ER=2 E_total R_total/(E_total+R_total+K+sqrt((E_total-R_total)^2+2K(E_total+R_total)+K²)).
        Stable algebra avoids subtracting two nearly equal numbers. All pool
        concentrations are mM; kcat(T) converts mM/s capacity into mM enzyme.
        No post-hoc cap or monotonicity enforcement is applied to assimilation.
        """
        kcat=self.kcat25*1.93**((temperature-25)/10)
        enzyme=vmax_c/kcat
        affinity=kr*regulation
        discriminant=(enzyme-rubp_total)**2+2*affinity*(enzyme+rubp_total)+affinity**2
        bound=2*enzyme*rubp_total/(enzyme+rubp_total+affinity+np.sqrt(discriminant))
        fraction=bound/enzyme
        vc=vmax_c*fraction*co2/(co2+kc*(1+oxygen/ko))
        # Preserve the supplied model's oxygenase/carboxylase capacity ratio.
        vo=.24*vmax_c*fraction*oxygen/(oxygen+ko*(1+co2/kc))
        return vc,vo,{'enzyme_mM':float(enzyme),'bound_RuBP_mM':float(bound),
                      'free_RuBP_mM':float(rubp_total-bound),'shared_fraction':float(fraction)}

    def respiration(self,temperature):
        return self.rd25*self.rd_q10**((temperature-25)/10)

    def coupled_gdc_reassimilation(self,gdc_potential,atp,gdc_capacity):
        """Quasi-steady NH3 removal: GDC output cannot exceed ATP-supported GS.

        This is a declared lumped pathway assumption, not ATP as a substrate
        of the GDC enzyme. Each NH3 costs one ATP (GS) and one NADPH equivalent
        (GOGAT); the latter is reported against the buffered reductant budget.
        GS reference capacity is provisionally set equal to GDC capacity, not
        fitted to leaf data. No NH3 pool is added to the 35-state model.
        """
        gs_capacity=gdc_capacity*atp/(self.km_atp_reassimilation+atp)
        return min(gdc_potential,gs_capacity)


# Carbon in sugar/amino-acid moieties; buffered gas and nucleotide skeletons
# are outside this fast network boundary. Rd uses a separate reserve pool.
CARBON_COUNTS=np.array([5,3,3,3,6,4,7,7,0,0,0,0,6,5,
                        3,2,2,2,2,3,2,3,3,3,6,6,6,0,0,0,6,0,12,12,3],dtype=float)


def carbon_residual(model,t,y):
    """mM C/s; explicitly account for CO2 fixation, PR and carbon exports."""
    dy=model.rhs(t,y);v=model.fluxes(t,y,area=False)
    boundary=v[0]-v[23]-6*v[12]-12*v[35]-3*v[39]
    return float(CARBON_COUNTS@dy-boundary)
