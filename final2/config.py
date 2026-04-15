"""
config.py  --  all constants shared across environment and algorithms
"""

import os

# -- Paths -------------------------------------------------------------------
SIM_DIR       = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'simulations'))
FORKLIFT_URDF = os.path.join(SIM_DIR, 'forklift_mast.urdf')

# -- Simulation --------------------------------------------------------------
SPEED        = 15
LEFT_WHEELS  = [2, 4]
RIGHT_WHEELS = [3, 5]
CAM_W, CAM_H = 320, 240
FOV          = 90
LIFT_JOINT   = 0
MAGNET_LINK  = 2
LIFT_STEP    = 0.02
LIFT_MIN, LIFT_MAX = -0.4, 0.4

# -- Object geometry ---------------------------------------------------------
BIG_R,  BIG_H = 0.7,  0.5
SML_R,  SML_H = 0.4 / 2.5, 0.5 / 2.5   # 0.16, 0.20

# -- Task phases -------------------------------------------------------------
PHASE_FIND_GREEN     = 0
PHASE_APPROACH_GREEN = 1
PHASE_PICKUP         = 2
PHASE_FIND_RED       = 3
PHASE_APPROACH_RED   = 4
PHASE_DROP           = 5
N_PHASES             = 6

PHASE_NAMES = ["find-G", "appr-G", "pickup", "find-R", "appr-R", "drop  "]

# -- Task thresholds ---------------------------------------------------------
GREEN_CLOSE_AREA   = 0.012
RED_CLOSE_AREA     = 0.120
MAGNET_DIST        = 0.5
AUTO_REVERSE_STEPS = 40

# -- Episode limits ----------------------------------------------------------
MAX_STEPS = 700

# -- Observation / action sizes ----------------------------------------------
OBS_DIM    = 10
N_ACTIONS  = 5

ACTION_NAMES = ["spin-L", "spin-R", "fwd   ", "fwd+L ", "fwd+R "]

# -- PPO hyperparameters -----------------------------------------------------
PPO = dict(
    MAX_EPISODES  = 2000,
    ROLLOUT_STEPS = 4096,   # ~6 full episodes per update; less noisy signal
    N_EPOCHS      = 10,
    MINI_BATCH    = 128,    # scaled with rollout
    GAMMA         = 0.99,
    GAE_LAMBDA    = 0.95,
    CLIP_EPS      = 0.2,
    ENTROPY_COEF  = 0.01,   # was 0.05 — too high, prevented convergence
    VALUE_COEF    = 0.5,
    LR            = 3e-4,
    GRAD_CLIP     = 0.5,
    EARLY_STOP_RATE = 0.85,
    HIDDEN        = 256,    # was 128 — match SAC capacity
    TARGET_KL     = 0.015,  # abort epoch early if policy drifts too far
)

# -- TQC hyperparameters -----------------------------------------------------
TQC = dict(
    MAX_EPISODES        = 2000,
    REPLAY_SIZE         = 100_000,
    BATCH_SIZE          = 256,
    GAMMA               = 0.99,
    LR_ACTOR            = 3e-4,
    LR_CRITIC           = 3e-4,
    LR_ALPHA            = 3e-5,
    ALPHA_INIT          = 0.2,
    TAU                 = 0.005,
    HIDDEN              = 256,
    EARLY_STOP_RATE     = 0.85,
    GRAD_CLIP           = 5.0,
    LOG_ALPHA_MIN       = -10.0,
    LOG_ALPHA_MAX       =  2.0,
    # Distributional settings
    N_QUANTILES         = 25,   # quantile atoms per critic
    N_CRITICS           = 5,    # independent critic networks
    TOP_DROP_PER_CRITIC = 2,    # atoms dropped per critic in target
                                # → kept = N_CRITICS * (N_QUANTILES - TOP_DROP_PER_CRITIC)
)

# -- SAC + PER hyperparameters -----------------------------------------------
SAC = dict(
    MAX_EPISODES   = 2000,
    REPLAY_SIZE    = 100_000,
    BATCH_SIZE     = 256,
    GAMMA          = 0.99,
    LR_ACTOR       = 3e-4,
    LR_CRITIC      = 3e-4,
    LR_ALPHA       = 3e-5,        # slower alpha adaptation prevents runaway
    ALPHA_INIT     = 0.2,
    TAU            = 0.005,
    HIDDEN         = 256,
    EARLY_STOP_RATE = 0.85,
    LOG_ALPHA_MIN  = -10.0,       # alpha floor ~4.5e-5  (never fully off)
    LOG_ALPHA_MAX  =  2.0,        # alpha cap  ~7.4      (hard clamp)
    GRAD_CLIP      = 5.0,         # critic gradient clip
    # -- PER (Schaul et al. 2015) --------------------------------------------
    PER_ALPHA      = 0.6,         # prioritisation exponent (0=uniform, 1=full)
    PER_BETA_START = 0.4,         # IS correction start (annealed -> 1.0)
    PER_BETA_END   = 1.0,         # IS correction end (fully unbiased)
    PER_EPS        = 1e-6,        # small constant added to |TD| to avoid zero priority
)
