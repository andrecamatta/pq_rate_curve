"""
constants.jl - Global constants for PQRateCurve module

This module centralizes all magic numbers and constants used throughout the codebase,
providing a single source of truth and improving maintainability.
"""

# ============================================================================
# Optimization Penalties
# ============================================================================

"""Penalty for discount factor validity violations (no-arbitrage conditions)"""
const PENALTY_DISCOUNT_VIOLATION = 1e9

"""Penalty for invalid parameters (NaN, Inf)"""
const PENALTY_INVALID_PARAMS = 1e10

"""Penalty for parameters near zero (τ values)"""
const PENALTY_NEAR_ZERO = 1000.0

"""Penalty for β₀ too high (unrealistic long-term rate > 25%)"""
const PENALTY_BETA0_TOO_HIGH = 10000.0

"""Penalty multiplier for SELIC rate constraint"""
const PENALTY_SELIC_MULTIPLIER = 1000000.0

"""Penalty for invalid pricing (high error placeholder)"""
const PENALTY_INVALID_PRICING = 1e12

"""Penalty value returned by NSS rate calculation when invalid"""
const NSS_INVALID_RATE_PENALTY = 1e9

# ============================================================================
# SELIC Rate Defaults
# ============================================================================

"""Default SELIC rate to use when API fails (10.5% p.a.)"""
const DEFAULT_SELIC_RATE = 0.105

"""Maximum number of days to look back when fetching SELIC from API"""
const SELIC_FALLBACK_DAYS = 10

# ============================================================================
# Time Conventions
# ============================================================================

"""Business days per year (Brazilian convention)"""
const BUSINESS_DAYS_PER_YEAR = 252

"""Day fraction for 1 business day (1/252)"""
const ONE_DAY_FRACTION = 1 / BUSINESS_DAYS_PER_YEAR

# ============================================================================
# Bayesian Optimization Safety
# ============================================================================

"""Safety multiplier for Bayesian optimization iteration limit (prevents runaway)"""
const BAYESIAN_SAFETY_MULTIPLIER = 20

# ============================================================================
# NSS Parameter Bounds
# ============================================================================

"""Minimum τ value to avoid numerical instability in NSS formula"""
const TAU_MIN_VALUE = 0.005

"""Minimum time value for NSS rate calculation (avoid division by zero)"""
const NSS_MIN_TIME = 1e-6

"""Maximum β₀ value (long-term rate cap at 25%)"""
const BETA0_MAX_VALUE = 0.25

"""Minimum rate allowed in validation (-2% for slight negative rates)"""
const MIN_ALLOWED_RATE = -0.02

"""Maximum rate allowed in validation (100% sanity check)"""
const MAX_ALLOWED_RATE = 1.0

# ============================================================================
# Default Values
# ============================================================================

"""Default minimum duration for bond weighting (fallback when duration <= 0)"""
const DEFAULT_MIN_DURATION = 0.1

"""Default minimum number of bonds required for curve fitting"""
const DEFAULT_MIN_BONDS = 3

"""Default bond quantity when not specified"""
const DEFAULT_BOND_QUANTITY = 1000.0

"""Face value for LTN (zero-coupon bonds)"""
const LTN_FACE_VALUE = 1000.0

"""Face value + final coupon for NTN-F (fixed-rate coupon bonds)"""
const NTNF_MATURITY_VALUE = 1050.0  # 1000 principal + 50 final coupon

"""Semi-annual coupon payment for NTN-F (5% = 10% p.a.)"""
const NTNF_COUPON_VALUE = 50.0
