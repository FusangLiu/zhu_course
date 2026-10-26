"""35-state C3 model, GPL-3.0-or-later. No MATLAB runtime required.
Original model copyright (2007) Xin-Guang Zhu and Stephen P. Long, University of Illinois. Python translation and declared physiological revision distributed under GPL-3.0-or-later; see COPYING."""
from .model import C3Model, simulate, STATE_NAMES, FLUX_NAMES, ENZYMES
from .physiology import PhysiologyParameters

__all__ = ['C3Model', 'simulate', 'STATE_NAMES', 'FLUX_NAMES', 'ENZYMES', 'PhysiologyParameters']
