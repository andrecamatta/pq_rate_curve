"""
high_level_api.jl - High-level API functions for common workflows

This module provides user-friendly functions that encapsulate complex workflows:
- fit_curves_for_period: Fit NSS curves for a date range
- create_yield_curve_animation: Generate video animation from fitted curves

These functions are designed for programmatic use while the CLI scripts provide
command-line interfaces to the same functionality.

Note: Hyperparameter validation is done via the CLI script run_continuous_walkforward_cv.jl
due to its complex distributed parallelization requirements.
"""

using CSV, DataFrames, Plots, Dates, Statistics, TOML
using Plots.Measures

# ============================================================================
# Data structures for curve fitting
# ============================================================================

"""
Result of fitting NSS curve for a single day.
"""
struct DayResult
    date::Date
    success::Bool
    params::Union{Vector{Float64}, Nothing}
    cost::Union{Float64, Nothing}
    n_bonds::Int
    outliers_removed::Int
    error_message::Union{String, Nothing}
    used_previous_params::Bool
end

# ============================================================================
# Helper functions for curve fitting
# ============================================================================

"""
    _normalize_config_format(config::Dict; base_config::Union{Dict,Nothing}=nothing) -> Dict{String, Any}

Normalize configuration format for compatibility between optimal_config.toml and config.toml.
If base_config is provided (from config.toml), uses it as fallback for missing values,
especially bounds which should always come from config.toml.
"""
function _normalize_config_format(config::Dict; base_config::Union{Dict,Nothing}=nothing)
    normalized = Dict{String, Any}()

    # Handle nested structure from optimal_config.toml
    actual_config = haskey(config, "optimal_config") ? config["optimal_config"] : config

    # Extract PSO parameters
    if haskey(actual_config, "pso")
        pso = actual_config["pso"]
        normalized["N"] = pso["N"]
        normalized["C1"] = pso["C1"]
        normalized["C2"] = pso["C2"]
        normalized["omega"] = pso["omega"]
        normalized["f_calls_limit"] = pso["f_calls_limit"]

        # Get bounds using centralized function (single source of truth)
        # Prefer base_config if available, otherwise use config defaults
        bounds_config = base_config !== nothing ? base_config : config
        lower_bounds, upper_bounds = get_pso_bounds(bounds_config)
        normalized["lower_bounds"] = lower_bounds
        normalized["upper_bounds"] = upper_bounds
    else
        error("Missing required configuration section: pso")
    end

    # Extract optimization parameters
    if haskey(actual_config, "optimization")
        opt = actual_config["optimization"]
        normalized["use_lbfgs"] = opt["use_lbfgs"]
        normalized["temporal_penalty_weight"] = opt["temporal_penalty_weight"]
    else
        error("Missing required configuration section: optimization")
    end

    # Extract outlier detection parameters
    if haskey(actual_config, "outlier_detection")
        outlier = actual_config["outlier_detection"]
        # New system: error_threshold_global and ultra_low_factor (independent filters)
        normalized["error_threshold_global"] = get(outlier, "error_threshold_global", 20.0)
        normalized["fator_liq"] = outlier["fator_liq"]
        normalized["ultra_low_factor"] = get(outlier, "ultra_low_factor", 3.0)
        # Old system (backward compatibility): if error_threshold_global missing, try mad_threshold
        if !haskey(outlier, "error_threshold_global") && haskey(outlier, "mad_threshold")
            normalized["error_threshold_global"] = outlier["mad_threshold"]
        end
    else
        error("Missing required configuration section: outlier_detection")
    end

    # Extract fit_curves parameters - prefer base_config if available
    if base_config !== nothing && haskey(base_config, "fit_curves")
        fit_curves_config = base_config["fit_curves"]
    else
        fit_curves_config = get(actual_config, "fit_curves", Dict())
    end

    normalized["continuity_search_days"] = get(fit_curves_config, "continuity_search_days", 30)
    normalized["min_bonds_for_fit"] = get(fit_curves_config, "min_bonds_for_fit", 3)
    normalized["lbfgs_max_iterations"] = get(fit_curves_config, "lbfgs_max_iterations", 50)

    return normalized
end

"""
    _read_optimal_config(config_file::String) -> Dict{String, Any}

Read and normalize configuration from TOML file.
Always loads config.toml as base (for bounds and structure), then overlays
optimal hyperparameters from optimal_config.toml if available.
"""
function _read_optimal_config(config_file::String)
    # Always load base config using ConfigService
    if !isfile(config_file)
        error("Base configuration file not found: $config_file")
    end

    base_config_service = load_config(config_file)
    base_config = get_raw_config(base_config_service)

    # Try to load optimal_config.toml for optimal hyperparameters
    optimal_path = joinpath(dirname(config_file), "optimal_config.toml")
    if isfile(optimal_path)
        # Use optimal hyperparameters but with bounds from base config
        optimal_config_service = load_config(optimal_path)
        optimal_config = get_raw_config(optimal_config_service)
        return _normalize_config_format(optimal_config; base_config=base_config)
    end

    # No optimal config found, use base config only
    return _normalize_config_format(base_config)
end

"""
    _find_continuity_params(start_date::Date, config::Dict; verbose::Bool=false) -> Union{Vector{Float64}, Nothing}

Search for continuity parameters by fitting curve for previous business day.
"""
function _find_continuity_params(start_date::Date, config::Dict{String, Any}; verbose::Bool=false)
    if verbose
        println("🔍 Buscando parâmetros de continuidade antes de $start_date...")
    end

    search_date = start_date - Day(1)

    for _ in 1:config["continuity_search_days"]
        if Dates.dayofweek(search_date) in 1:5  # Business day
            try
                if verbose
                    println("   Tentando $search_date...")
                end

                result = _fit_nss_single_day(search_date, config, nothing; verbose=false)
                if result.success
                    if verbose
                        println("✅ Parâmetros encontrados em $search_date")
                    end
                    return result.params
                end
            catch
                # Continue searching
            end
        end
        search_date -= Day(1)
    end

    if verbose
        println("⚠️ Não foi possível encontrar parâmetros de continuidade")
    end
    return nothing
end

"""
    _fit_nss_single_day(date::Date, config::Dict, previous_params; verbose::Bool=true) -> DayResult

Fit NSS curve for a single date with outlier removal and optional L-BFGS refinement.
Uses fixed threshold outlier detection and ultra-low liquidity filter.
"""
function _fit_nss_single_day(date::Date, config::Dict{String, Any}, previous_params::Union{Vector{Float64}, Nothing};
                             verbose::Bool=true)
    try
        df = load_bacen_data(date, date)

        if nrow(df) < config["min_bonds_for_fit"]
            return DayResult(date, false, nothing, nothing, 0, 0, "Dados insuficientes (<$(config["min_bonds_for_fit"]) títulos)", false)
        end

        cash_flows, bond_quantities, _ = generate_cash_flows_with_quantity(df, date)

        if length(cash_flows) < config["min_bonds_for_fit"]
            return DayResult(date, false, nothing, nothing, 0, 0, "Cash flows insuficientes (<$(config["min_bonds_for_fit"]))", false)
        end

        if verbose
            print("📊 $date: $(length(cash_flows)) títulos")
        end

        # Optimization with outlier removal (now using fixed threshold + ultra-low liquidity filter)
        params, cost, final_cash_flows, outliers_removed, iterations = optimize_nelson_siegel_svensson_with_mad_outlier_removal(
            cash_flows, date, config["lower_bounds"], config["upper_bounds"];
            previous_params=previous_params,
            temporal_penalty_weight=config["temporal_penalty_weight"],
            pso_N=config["N"],
            pso_C1=config["C1"],
            pso_C2=config["C2"],
            pso_omega=config["omega"],
            pso_f_calls_limit=config["f_calls_limit"],
            error_threshold_global=config["error_threshold_global"],
            fator_liq=config["fator_liq"],
            ultra_low_factor=config["ultra_low_factor"],
            bond_quantities=bond_quantities,
            verbose=false
        )

        # L-BFGS refinement if configured
        if config["use_lbfgs"]
            try
                params_lbfgs, cost_lbfgs, lbfgs_success = refine_nss_with_lbfgs(
                    final_cash_flows, date, params, config["lower_bounds"], config["upper_bounds"];
                    max_iterations=config["lbfgs_max_iterations"], show_trace=false,
                    previous_params=previous_params,
                    temporal_penalty_weight=config["temporal_penalty_weight"],
                    verbose=false
                )

                if lbfgs_success && cost_lbfgs < cost
                    params = params_lbfgs
                    cost = cost_lbfgs
                    if verbose
                        print(" + L-BFGS")
                    end
                end
            catch
                # Keep PSO if L-BFGS fails
            end
        end

        used_previous = previous_params !== nothing

        if verbose
            outlier_info = outliers_removed > 0 ? " (-$outliers_removed títulos)" : ""
            println(" → ✅ Custo final: $(round(cost, digits=4))$outlier_info")
        end

        return DayResult(date, true, params, cost, length(final_cash_flows), outliers_removed, nothing, used_previous)

    catch e
        error_msg = string(e)
        if verbose
            println(" → ❌ $error_msg")
        end
        return DayResult(date, false, nothing, nothing, 0, 0, error_msg, false)
    end
end

"""
    _save_results_to_csv(results::Vector{DayResult}, config::Dict, output_base::String) -> String

Save curve fitting results to CSV file with timestamp.
"""
function _save_results_to_csv(results::Vector{DayResult}, config::Dict{String, Any}, output_base::String)
    timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    filename = "$(output_base)_$(timestamp).csv"

    data = DataFrame(
        Data = Date[],
        Sucesso = Bool[],
        Beta0 = Union{Float64, Missing}[],
        Beta1 = Union{Float64, Missing}[],
        Beta2 = Union{Float64, Missing}[],
        Beta3 = Union{Float64, Missing}[],
        Tau1 = Union{Float64, Missing}[],
        Tau2 = Union{Float64, Missing}[],
        Custo = Union{Float64, Missing}[],
        NumTitulos = Int[],
        OutliersRemovidos = Int[],
        UsouPreviousParams = Bool[],
        ErroMensagem = Union{String, Missing}[]
    )

    for result in results
        if result.success && result.params !== nothing
            push!(data, (
                result.date, result.success,
                result.params[1], result.params[2], result.params[3],
                result.params[4], result.params[5], result.params[6],
                result.cost, result.n_bonds, result.outliers_removed,
                result.used_previous_params,
                missing
            ))
        else
            push!(data, (
                result.date, result.success,
                missing, missing, missing, missing, missing, missing,
                missing, result.n_bonds, result.outliers_removed,
                result.used_previous_params,
                result.error_message
            ))
        end
    end

    sort!(data, :Data)
    CSV.write(filename, data)

    return filename
end

# ============================================================================
# Main high-level API functions
# ============================================================================

"""
    fit_curves_for_period(start_date::Date, end_date::Date;
                         config_file::String="config.toml",
                         output_csv::Union{String,Nothing}="curvas_nss",
                         find_continuity::Bool=true,
                         verbose::Bool=true) -> (Vector{DayResult}, Dict{String, Any})

Fit NSS curves for all business days in a given period.

# Arguments
- `start_date`: First date to fit curves for
- `end_date`: Last date to fit curves for

# Options
- `config_file`: Path to configuration TOML file (default: "config.toml")
- `output_csv`: Base name for output CSV file, or `nothing` to skip saving (default: "curvas_nss")
- `find_continuity`: Search for previous parameters for temporal continuity (default: true)
- `verbose`: Print progress information (default: true)

# Returns
- `results`: Vector of DayResult structs with fitting results for each date
- `config`: Configuration dictionary used for fitting

# Example
```julia
using PQRateCurve, Dates

# Fit curves for Q1 2024
results, config = fit_curves_for_period(
    Date(2024, 1, 1),
    Date(2024, 3, 31);
    output_csv="curvas_q1_2024",
    verbose=true
)

# Check results
successful = sum(r.success for r in results)
println("Successfully fitted \$successful/\$(length(results)) curves")

# Access individual results
for r in results[1:5]
    if r.success
        println("\$(r.date): β₀=\$(round(r.params[1], digits=4)), cost=\$(round(r.cost, digits=6))")
    end
end
```
"""
function fit_curves_for_period(start_date::Date, end_date::Date;
                               config_file::String="config.toml",
                               output_csv::Union{String,Nothing}="curvas_nss",
                               find_continuity::Bool=true,
                               verbose::Bool=true)

    if verbose
        println("📊 FIT SEQUENCIAL DE CURVAS NSS")
        println("🔗 COM CONTINUIDADE TEMPORAL")
        println("=" ^ 50)
    end

    # Load configuration
    config = _read_optimal_config(config_file)

    if verbose
        println("✅ Configuração ótima:")
        println("   N=$(config["N"]), C1=$(round(config["C1"], digits=2)), C2=$(round(config["C2"], digits=2))")
        println("   ω=$(round(config["omega"], digits=2)), L-BFGS=$(config["use_lbfgs"])")
        println("   Temporal penalty=$(round(config["temporal_penalty_weight"], digits=4))")
        println("   Error threshold (global)=$(round(config["error_threshold_global"], digits=2))")
        println("   Ultra-low liquidity factor=$(round(config["ultra_low_factor"], digits=2))")
        println("   Fator liquidez=$(round(config["fator_liq"], digits=4)) ($(round(config["fator_liq"]*100, digits=2))% do total)")
    end

    # Generate dates (using BusinessDays.jl with Brazilian calendar)
    all_dates = get_business_dates(start_date, end_date)
    total_dates = length(all_dates)

    if verbose
        println("\n📅 Período: $start_date a $end_date")
        println("📊 Datas úteis: $total_dates")
        println("⏱️ Estimativa: $(round(total_dates * 2 / 60, digits=1)) minutos")
    end

    # Search for continuity parameters
    previous_params = nothing
    if find_continuity
        previous_params = _find_continuity_params(start_date, config; verbose=verbose)
    end

    if verbose
        println("\n🚀 Iniciando fit sequencial...")
    end

    # Sequential processing
    start_time = time()
    all_results = DayResult[]
    current_previous_params = previous_params
    successful_fits = 0

    for (i, date) in enumerate(all_dates)
        if verbose && (i % 50 == 1 || i == total_dates)
            progress_pct = round(i/total_dates*100, digits=1)
            println("🔄 Progresso: $i/$total_dates ($progress_pct%)")
        end

        result = _fit_nss_single_day(date, config, current_previous_params; verbose=verbose)
        push!(all_results, result)

        if result.success
            current_previous_params = result.params
            successful_fits += 1
        end
    end

    elapsed_time = time() - start_time

    if verbose
        println("\n⏱️ Processamento completo em $(round(elapsed_time/60, digits=1)) minutos")

        continuity_count = sum(r.used_previous_params for r in all_results if r.success)
        println("🔗 Continuidade: $continuity_count/$successful_fits fits usaram previous_params")
        println("📊 Taxa de sucesso: $(round(successful_fits/total_dates*100, digits=1))%")
    end

    # Save to CSV if requested
    if output_csv !== nothing && !isempty(all_results)
        filename = _save_results_to_csv(all_results, config, output_csv)

        # Print summary
        successful = sum(r.success for r in all_results)
        println("\n💾 Dados salvos: $filename")
        println("📊 Resumo: $successful/$(length(all_results)) sucessos ($(round(successful/length(all_results)*100, digits=1))%)")
    end

    return all_results, config
end

"""
    create_yield_curve_animation(csv_file::String, output_video::String;
                                  fps::Int=10,
                                  duration::Int=30,
                                  maturities::Vector{Float64}=[0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0],
                                  plot_size::Tuple{Int,Int}=(1200, 800),
                                  plot_dpi::Int=200,
                                  config_file::Union{String,Nothing}=nothing) -> String

Create an animated video of yield curves from CSV file with NSS parameters.

# Arguments
- `csv_file`: Path to CSV file with fitted NSS parameters (output from fit_curves_for_period)
- `output_video`: Path for output MP4 video file

# Options
- `fps`: Frames per second (default: 10)
- `duration`: Video duration in seconds (default: 30)
- `maturities`: Vector of maturities in years to plot (default: 0.25 to 10.0 years)
- `plot_size`: Tuple (width, height) for plot size (default: (1200, 800))
- `plot_dpi`: Plot DPI resolution (default: 200)
- `config_file`: Optional path to config.toml to override defaults (default: none)

# Returns
- Path to generated video file

# Example
```julia
using PQRateCurve

# Create animation from fitted curves
video_path = create_yield_curve_animation(
    "curvas_nss_2024-01-01_12-00-00.csv",
    "animation_2024.mp4";
    fps=15,
    duration=20
)
println("Video created: \$video_path")
```
"""
function create_yield_curve_animation(csv_file::String, output_video::String;
                                       fps::Int=10,
                                       duration::Int=30,
                                       maturities::Vector{Float64}=[0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0],
                                       plot_size::Tuple{Int,Int}=(1200, 800),
                                       plot_dpi::Int=200,
                                       config_file::Union{String,Nothing}=nothing)

    # Load configuration if provided using ConfigService
    if config_file !== nothing && isfile(config_file)
        config_service = load_config(config_file)
        config = get_raw_config(config_service)
        anim_config = get(config, "animation", Dict())
        fps = get(anim_config, "fps", fps)
        duration = get(anim_config, "duration", duration)
        plot_dpi = get(anim_config, "plot_dpi", plot_dpi)
        plot_size = tuple(get(anim_config, "plot_size", collect(plot_size))...)
        maturities = get(anim_config, "maturities", maturities)
    end

    # Set plotting backend
    gr()

    # Load the data
    println("📊 Loading yield curve data from $csv_file...")
    df = CSV.read(csv_file, DataFrame)

    # Filter successful fits
    successful_fits = df[df.Sucesso .== true, :]
    dates = successful_fits.Data
    sort!(dates)

    if isempty(dates)
        error("No successful curve fits found in $csv_file")
    end

    # Create time points for animation
    time_points = range(1, length(dates), length=Int(round(fps * duration)))

    # Find global min and max rates for consistent y-axis
    all_rates = Float64[]
    for row in eachrow(successful_fits)
        rates = [nss_rate(m, [row.Beta0, row.Beta1, row.Beta2, row.Beta3, row.Tau1, row.Tau2]) for m in maturities] .* 100
        append!(all_rates, rates)
    end
    min_rate = minimum(all_rates)
    max_rate = maximum(all_rates)
    rate_range = max_rate - min_rate
    y_min = max(0, min_rate - 0.1 * rate_range)
    y_max = max_rate + 0.1 * rate_range

    # Prepare animation
    println("🎬 Creating animation...")
    anim = @animate for i in time_points
        # Get the closest date index
        idx = min(round(Int, i), length(dates))
        date = dates[idx]

        # Get data for this date
        date_data = successful_fits[successful_fits.Data .== date, :][1, :]

        # Calculate yield curve
        rates = [nss_rate(m, [date_data.Beta0, date_data.Beta1, date_data.Beta2, date_data.Beta3, date_data.Tau1, date_data.Tau2]) for m in maturities] .* 100

        # Create the plot
        p = plot(
            maturities, rates,
            xlabel="Prazo (anos)",
            ylabel="Taxa (%)",
            title="Curvas de Juros - Títulos Públicos Brasileiros",
            legend=false,
            linewidth=3,
            linecolor=:blue,
            size=plot_size,
            dpi=plot_dpi,
            ylims=(y_min, y_max),
            xlims=(0, maximum(maturities)),
            xticks=0:1:maximum(maturities),
            grid=true,
            gridstyle=:dot,
            gridalpha=0.3,
            framestyle=:box,
            margin=5mm
        )

        # Add date annotation
        annotate!(p, maximum(maturities) * 0.02, y_max * 0.95,
                 text("Data: $(Dates.format(date, "dd/mm/yyyy"))", :left, 12))

        # Add parameter annotations
        param_text = "β₀=$(round(date_data.Beta0, digits=4))  β₁=$(round(date_data.Beta1, digits=4))  " *
                    "β₂=$(round(date_data.Beta2, digits=4))  β₃=$(round(date_data.Beta3, digits=4))\\n" *
                    "τ₁=$(round(date_data.Tau1, digits=2))  τ₂=$(round(date_data.Tau2, digits=2))  " *
                    "Títulos=$(date_data.NumTitulos)"
        annotate!(p, maximum(maturities) * 0.02, y_min + (y_max - y_min) * 0.05,
                 text(param_text, :left, 8, :gray))

        p
    end

    # Save animation
    println("🎥 Saving animation to $output_video...")
    Plots.mp4(anim, output_video, fps=fps)

    # Summary
    println("✅ Animation complete! Video saved as $output_video")
    println("📊 Summary:")
    println("   - Input file: $csv_file")
    println("   - Output video: $output_video")
    println("   - Duration: $duration seconds")
    println("   - FPS: $fps")
    println("   - Total successful curves: $(length(dates))")
    println("   - Date range: $(minimum(dates)) to $(maximum(dates))")
    println("   - First successful date: $(minimum(dates))")
    println("   - Last successful date: $(maximum(dates))")

    return output_video
end
