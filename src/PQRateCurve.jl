"""
PQRateCurve.jl - Brazilian Government Bond Yield Curve Fitting Module

This module provides functionality for fitting Nelson-Siegel-Svensson yield curves
to Brazilian government bond data using Particle Swarm Optimization and 
Levenberg-Marquardt refinement.

Main features:
- Nelson-Siegel-Svensson yield curve estimation
- Outlier detection using MAD and liquidity criteria  
- PSO+LM hybrid optimization
- Walk-forward cross-validation
- Temporal continuity constraints
"""
module PQRateCurve

using Dates, Statistics, DataFrames, CSV, HTTP, ZipFile, LinearAlgebra
using Random, Optim, TOML, JSON

# Export main functionality
export 
    # Financial math functions
    nss_rate, price_bond, calculate_ytm, calculate_duration, yearfrac,
    
    # Data handling
    load_bacen_data, generate_cash_flows_with_quantity, load_configuration,
    save_optimal_configuration,
    
    # Outlier detection  
    detect_outliers_mad_and_liquidity, calculate_mad,
    
    # Optimization and estimation
    optimize_nelson_siegel_svensson_with_mad_outlier_removal,
    refine_nss_with_levenberg_marquardt, calculate_pricing_error_duration_only,
    calculate_out_of_sample_cost_reais, precompute_cash_flow_times,
    
    # Walk-forward validation
    run_walkforward_validation, generate_pso_configs, generate_focused_pso_configs

# Include all module files
include("financial_math.jl")
include("data_handling.jl") 
include("outlier_detection.jl")
include("estimation.jl")

end # module PQRateCurve