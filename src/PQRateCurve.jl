"""
PQRateCurve.jl - Brazilian Government Bond Yield Curve Fitting Module

This module provides functionality for fitting Nelson-Siegel-Svensson yield curves
to Brazilian government bond data using Particle Swarm Optimization and 
L-BFGS refinement.

Main features:
- Nelson-Siegel-Svensson yield curve estimation
- Outlier detection using MAD and liquidity criteria  
- PSO+L-BFGS hybrid optimization
- Walk-forward cross-validation
- Temporal continuity constraints
"""
module PQRateCurve

using Dates, Statistics, DataFrames, CSV, HTTP, ZipFile, LinearAlgebra, BusinessDays
using Random, Optim, TOML, JSON, SQLite

# Export main functionality
export
    # Financial math functions
    nss_rate, price_bond, calculate_ytm, calculate_duration, yearfrac,

    # Data handling
    load_bacen_data, generate_cash_flows_with_quantity, load_configuration,
    save_optimal_configuration, get_business_dates, get_pso_bounds,

    # Configuration management
    ConfigService, load_config, get_cv_config, get_pso_config,
    get_hyperparams_config, get_raw_config, default_config, reset_default_config!,

    # Formatting utilities
    format_percentage, format_score, format_cost, format_basis_points,
    format_nss_params, format_pso_coefficient, format_temporal_penalty,
    format_overfitting_ratio, format_time_minutes, format_time_seconds,
    format_currency, format_quantity, format_error_threshold, format_liquidity_factor,

    # Cache management
    clear_cache, migrate_existing_cache,

    # Outlier detection
    detect_outliers_mad_and_liquidity, calculate_mad,

    # Optimization and estimation
    fit_nss,
    refine_nss_with_lbfgs, calculate_pricing_error_duration_only,
    calculate_out_of_sample_cost_reais, precompute_cash_flow_times,
    normalize_cost_by_volume,

    # Walk-forward validation
    run_walkforward_validation, generate_pso_configs, generate_focused_pso_configs,

    # High-level API functions
    fit_curves_for_period, create_yield_curve_animation, DayResult,

    # Persistence (SQLite database)
    init_database, save_curve, save_curve_failure,
    load_curves, load_curve, curve_exists,
    get_missing_dates, get_database_stats, get_rate,

    # Cortes de Selic precificados pela curva (WIRP)
    CopomMeeting, load_copom_calendar, current_selic, next_meetings,
    effective_date, implied_selic_path,

    # Curva com degraus nas datas de reunião (bootstrap)
    ZeroObservation, MeetingCurve, ltn_zero_rates, bootstrap_meeting_curve,
    implied_path, zero_rate_curve,

    # Curvas de referência da B3 (curva DI)
    fetch_b3_curve, b3_curve_observations, B3_REFERENCE_CURVES,
    interpolate_flat_forward

# Include all module files
include("constants.jl")
include("config_service.jl")
include("formatting.jl")
include("financial_math.jl")
include("data_handling.jl")
include("outlier_detection.jl")
include("estimation.jl")
include("persistence.jl")
include("high_level_api.jl")
include("copom.jl")
include("meeting_curve.jl")
include("b3_curves.jl")

end # module PQRateCurve