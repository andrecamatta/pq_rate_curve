"""
estimation.jl - Optimization and estimation functionality

This module contains the core optimization algorithms for Nelson-Siegel-Svensson
yield curve estimation, including PSO and Levenberg-Marquardt methods.
"""

using Random, HTTP, JSON, LinearAlgebra, Statistics, Optim, Metaheuristics

# --- Funções de Otimização e Estimação ---

"""
    calculate_nss_cost(params, cash_flows_with_times, selic_rate, previous_params, temporal_penalty_weight) -> Float64

Calculates the unified cost for a given set of NSS parameters.

This function is the single source of truth for the cost metric, used by
both PSO and LM optimizers to ensure consistency.

The cost includes:
- Weighted pricing error of the bond portfolio.
- Penalty for violating no-arbitrage conditions.
- Penalty for being outside of reasonable parameter bounds.
- Penalty for deviating from the daily SELIC rate.
- Penalty for lack of temporal continuity from previous day's parameters.
"""
function calculate_nss_cost(params::Vector{Float64},
                            cash_flows_with_times::Vector,
                            selic_rate::Float64,
                            previous_params::Union{Vector{Float64}, Nothing},
                            temporal_penalty_weight::Float64)
    # Penalty #1: No-arbitrage discount factor validity
    if !validate_discount_factors(params)
        return PENALTY_DISCOUNT_VIOLATION
    end

    # Penalty for NaN, Inf, or extreme parameters
    if any(isnan, params) || any(isinf, params)
        return PENALTY_INVALID_PARAMS
    end

    # Parameter bounds penalty (soft constraint for values near zero)
    penalty = 0.0
    if abs(params[5]) < TAU_MIN_VALUE || abs(params[6]) < TAU_MIN_VALUE
        penalty += PENALTY_NEAR_ZERO
    end

    # Penalidade por β0 muito baixo (taxa de longo prazo irrealista)
    # β0 > 25% (0.25) implica yield de longo prazo > 25% - improvável no Brasil
    if params[1] > BETA0_MAX_VALUE
        penalty += PENALTY_BETA0_TOO_HIGH
    end

    # SELIC constraint penalty: r(1 day) should be close to SELIC
    model_rate_1day = nss_rate(ONE_DAY_FRACTION, params)
    selic_penalty = PENALTY_SELIC_MULTIPLIER * (model_rate_1day - selic_rate)^2

    # Temporal continuity penalty (functional style)
    temporal_penalty = 0.0
    if previous_params !== nothing
        temporal_vertices = [0.5, 1.0, 3.0, 5.0, 10.0]
        temporal_penalty = temporal_penalty_weight * sum((nss_rate(t, params) - nss_rate(t, previous_params))^2 for t in temporal_vertices)
    end

    # Pricing error (functional style)
    pricing_errors = map(cash_flows_with_times) do (market_price, cf_with_times)
        isempty(cf_with_times) && return (0.0, 0.0)

        theoretical_price = sum(amount * exp(-nss_rate(t, params) * t) for (t, amount) in cf_with_times if t > 0)

        (isnan(theoretical_price) || isinf(theoretical_price) || theoretical_price <= 0.0) && return (PENALTY_INVALID_PRICING, 1.0) # Return high cost and weight

        duration = theoretical_price > 0 ? sum(t * amount * exp(-nss_rate(t, params) * t) for (t, amount) in cf_with_times if t > 0) / theoretical_price : DEFAULT_MIN_DURATION
        weight = 1.0 / sqrt(max(duration, DEFAULT_MIN_DURATION))

        error_sq = (theoretical_price - market_price)^2
        return (weight * error_sq, weight)
    end

    total_weighted_cost = sum(e[1] for e in pricing_errors)
    total_weight = sum(e[2] for e in pricing_errors)

    cost_bonds = total_weight > 0 ? total_weighted_cost / total_weight : PENALTY_INVALID_PRICING

    return cost_bonds + penalty + selic_penalty + temporal_penalty
end


# Note: The functions remove_outliers and the original detect_outliers_mad_and_liquidity
# have been removed from this file to eliminate duplication.
# They are now centralized in outlier_detection.jl.

# Global cache for SELIC rates
const SELIC_CACHE = Dict{Date, Float64}()
const SELIC_CACHE_PATH = joinpath(@__DIR__, "..", "raw", "selic_cache.json")
const SELIC_CACHE_LOADED = Ref(false)

# Helper to load the SELIC cache from a persistent file
function _load_selic_cache()
    if !isfile(SELIC_CACHE_PATH)
        return # No cache file yet
    end
    try
        json_str = read(SELIC_CACHE_PATH, String)
        if isempty(json_str)
            return
        end
        cached_data = JSON.parse(json_str)
        for (date_str, rate) in cached_data
            date = Date(date_str, "yyyy-mm-dd")
            SELIC_CACHE[date] = rate
        end
    catch e
        @warn "Could not load SELIC cache from $(SELIC_CACHE_PATH): $e"
    end
end

# Helper to save the SELIC cache to a persistent file
function _save_selic_cache()
    try
        # Prepare data with string keys for JSON compatibility
        to_save = Dict{String, Float64}(Dates.format(d, "yyyy-mm-dd") => r for (d, r) in SELIC_CACHE)
        open(SELIC_CACHE_PATH, "w") do f
            JSON.print(f, to_save, 4) # Use 4-space indent for readability
        end
    catch e
        @warn "Could not save SELIC cache to $(SELIC_CACHE_PATH): $e"
    end
end


# Fetches a single SELIC rate from the BACEN API for a given date.
function _fetch_selic_from_api(date::Date; verbose::Bool=true)
    date_str = Dates.format(date, "dd/mm/yyyy")
    url = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.432/dados?formato=json&dataInicial=$(date_str)&dataFinal=$(date_str)"

    try
        if verbose; @info "Fetching SELIC (series 432) from BACEN for $(date)..."; end
        headers = [
            "User-Agent" => "Mozilla/5.0 (compatible; Julia HTTP client)",
            "Accept" => "application/json"
        ]
        response = HTTP.get(url, headers; readtimeout=15, retries=3)  # Added headers for BACEN API

        content_type = HTTP.header(response, "Content-Type", "")
        if !contains(content_type, "application/json")
            body_preview = String(response.body)[1:min(100, end)]
            @warn "Unexpected content type for $(date): $(content_type). Body: '$(body_preview)'"
            return nothing
        end

        data = JSON.parse(String(response.body))

        if !isempty(data)
            selic_value = data[1]["valor"]
            return parse(Float64, selic_value) / 100.0
        end
    catch e
        @warn "Error fetching SELIC for $(date): $(e)."
    end

    return nothing
end

"""
    get_selic_rate(date::Date) -> Float64

Get SELIC rate for a specific date, using a persistent cache for performance.
"""
function get_selic_rate(date::Date; max_fallback_days=10, verbose::Bool=true)
    # 1. Load persistent cache if not already loaded in this session
    if !SELIC_CACHE_LOADED[]
        _load_selic_cache()
        SELIC_CACHE_LOADED[] = true
    end

    # 2. Check in-memory cache first
    if haskey(SELIC_CACHE, date)
        if verbose; @info "Cache hit for SELIC on $(date): $(round(SELIC_CACHE[date]*100, digits=2))% a.a."; end
        return SELIC_CACHE[date]
    end

    # 3. If not in cache, try to fetch from API with fallback
    for i in 0:max_fallback_days
        current_date = date - Day(i)
        
        # Check cache again for the fallback date
        if haskey(SELIC_CACHE, current_date)
            rate = SELIC_CACHE[current_date]
            if verbose; @info "Fallback cache hit on $(current_date) for original date $(date)."; end
            SELIC_CACHE[date] = rate # Cache for the original date to speed up future lookups
            return rate
        end

        # Fetch from API
        rate = _fetch_selic_from_api(current_date; verbose=verbose)
        
        if rate !== nothing
            if verbose; @info "API success for $(current_date): $(round(rate*100, digits=2))% a.a."; end
            # Update cache for both dates and save to file
            SELIC_CACHE[current_date] = rate
            SELIC_CACHE[date] = rate
            _save_selic_cache()
            return rate
        end
        if verbose; @info "API fetch failed for $(current_date), trying previous day..."; end
    end

    # 4. If API fails, use the most recent value from the cache
    if !isempty(SELIC_CACHE)
        latest_date = maximum(keys(SELIC_CACHE))
        rate = SELIC_CACHE[latest_date]
        @warn "API fetch failed after $(max_fallback_days) days. Using most recent cached rate from $(latest_date)."
        return rate
    end
    
    # 5. Absolute fallback if API and cache fail
    @error "CRITICAL: SELIC rate fetch failed completely and cache is empty. Using default rate: $(round(DEFAULT_SELIC_RATE*100, digits=2))% a.a."
    return DEFAULT_SELIC_RATE
end


"""
    optimize_pso_nss(cash_flows, ref_date; 
                    previous_params=nothing, temporal_penalty_weight=0.01,
                    pso_N=50, pso_C1=2.0, pso_C2=2.0, pso_omega=0.5, pso_f_calls_limit=1500,
                    use_liquidity_weights=false) -> (Vector{Float64}, Float64)

Optimize Nelson-Siegel-Svensson parameters using Particle Swarm Optimization.

Parameters:
- cash_flows: Vector of (market_price, cash_flow) tuples
- ref_date: Reference date
- previous_params: Previous day parameters for temporal continuity
- temporal_penalty_weight: Weight for temporal penalty
- pso_N: Number of particles
- pso_C1, pso_C2: PSO acceleration coefficients
- pso_omega: PSO inertia weight
- pso_f_calls_limit: Maximum function evaluations

Returns:
- best_params: Optimal NSS parameters
- best_cost: Optimal cost value
"""
function optimize_pso_nss(cash_flows, ref_date, lower_bounds::Vector{Float64}, upper_bounds::Vector{Float64};
                         previous_params=nothing, temporal_penalty_weight=0.01,
                         pso_N=50, pso_C1=2.0, pso_C2=2.0, pso_omega=0.5, pso_f_calls_limit=1500,
                         verbose::Bool=true)

    if isempty(cash_flows)
        return zeros(6), Inf
    end

    # Get SELIC rate for constraint
    selic_rate = get_selic_rate(ref_date; verbose=verbose)

    # Pre-calculate time fractions for performance
    cash_flows_with_times = []
    for (market_price, cash_flow) in cash_flows
        cf_with_times = [(yearfrac(ref_date, date), amount) for (date, amount) in cash_flow]
        push!(cash_flows_with_times, (market_price, cf_with_times))
    end

    # Wrapper for the unified cost function
    objective(params) = calculate_nss_cost(params, cash_flows_with_times, selic_rate, previous_params, temporal_penalty_weight)

    # Create options with function calls limit
    options = Options(f_calls_limit=pso_f_calls_limit, verbose=verbose, store_convergence=false)
    
    # Define the PSO algorithm with parameters from config and options
    pso_algorithm = PSO(N = pso_N, C1 = pso_C1, C2 = pso_C2, ω = pso_omega, options = options)

    # Run the optimization
    bounds = Metaheuristics.boxconstraints(lb=lower_bounds, ub=upper_bounds)
    result = Metaheuristics.optimize(objective, bounds, pso_algorithm)

    return minimizer(result), minimum(result)
end

"""
    refine_nss_with_levenberg_marquardt(cash_flows, ref_date, initial_params;
                                       max_iterations=50, show_trace=false,
                                       previous_params=nothing, temporal_penalty_weight=0.01) -> (Vector{Float64}, Float64, Bool)

Refine NSS parameters using Levenberg-Marquardt algorithm.

Parameters:
- cash_flows: Vector of (market_price, cash_flow) tuples
- ref_date: Reference date
- initial_params: Initial parameter guess (usually from PSO)
- max_iterations: Maximum LM iterations
- show_trace: Whether to show optimization trace
- previous_params: Previous day parameters for temporal continuity
- temporal_penalty_weight: Weight for temporal penalty

Returns:
- refined_params: Refined NSS parameters
- final_cost: Final cost value
- success: Whether optimization converged successfully
"""
function refine_nss_with_lbfgs(cash_flows, ref_date, pso_params, lower_bounds::Vector{Float64}, upper_bounds::Vector{Float64};
                                             max_iterations=100,
                                             show_trace=false,
                                             previous_params=nothing,
                                             temporal_penalty_weight=0.01,
                                             verbose::Bool=true)

    if verbose
        println("🔧 Aplicando refinamento com L-BFGS via Optim.jl...")
    end

    selic_rate = get_selic_rate(ref_date; verbose=verbose)

    # Pre-calculate time fractions for performance, same as in PSO
    cash_flows_with_times = []
    for (market_price, cash_flow) in cash_flows
        cf_with_times = [(yearfrac(ref_date, date), amount) for (date, amount) in cash_flow]
        push!(cash_flows_with_times, (market_price, cf_with_times))
    end

    # The objective function is the unified cost function
    objective(params) = calculate_nss_cost(params, cash_flows_with_times, selic_rate, previous_params, temporal_penalty_weight)

    try
        # Expand bounds slightly to avoid boundary warnings in L-BFGS
        # PSO parameters might be exactly on the boundary, but L-BFGS needs interior positions
        bound_expansion = 0.001  # Small expansion factor
        range_sizes = upper_bounds .- lower_bounds
        expanded_lower = lower_bounds .- bound_expansion .* range_sizes
        expanded_upper = upper_bounds .+ bound_expansion .* range_sizes
        
        # Use Optim.jl with a bounded optimizer (L-BFGS in a box)
        result = Optim.optimize(objective, expanded_lower, expanded_upper, pso_params,
                                Fminbox(LBFGS()),
                                Optim.Options(iterations = max_iterations, show_trace = show_trace))
        
        params_lbfgs = Optim.minimizer(result)
        cost_lbfgs = Optim.minimum(result)
        converged = Optim.converged(result)

        if verbose
            println("   Convergiu com L-BFGS: $(converged ? "✅" : "❌")")
            println("   Custo L-BFGS (unificado): $(round(cost_lbfgs, digits=6))")
        end
        
        return params_lbfgs, cost_lbfgs, converged
        
    catch e
        @warn "L-BFGS optimization failed: $e. Retornando parâmetros PSO."
        
        # Fallback to calculate original PSO cost for comparison
        pso_cost = objective(pso_params)
        
        return pso_params, pso_cost, false
    end
end

"""
    optimize_nelson_siegel_svensson_with_mad_outlier_removal(cash_flows, ref_date;
                                                           previous_params=nothing,
                                                           temporal_penalty_weight=0.01,
                                                           pso_N=50, pso_C1=2.0, pso_C2=2.0,
                                                           pso_omega=0.5, pso_f_calls_limit=1500,
                                                           error_threshold_global=20.0, fator_liq=0.015,
                                                           ultra_low_factor=3.0,
                                                           max_iterations=5, min_bonds=3,
                                                           bond_quantities=nothing) -> (Vector{Float64}, Float64, Vector, Int, Int)

Main optimization function with iterative outlier removal using FIXED THRESHOLD and liquidity criteria.

NEW APPROACH (replaces MAD-based):
1. First, remove ultra-low liquidity bonds (qty < fator_liq/ultra_low_factor) - independent of error
2. Then, iteratively remove high-error low-liquidity bonds (error > threshold AND qty < fator_liq)

Parameters:
- cash_flows: Vector of (market_price, cash_flow) tuples
- ref_date: Reference date
- previous_params: Previous day parameters for temporal continuity
- temporal_penalty_weight: Weight for temporal penalty
- pso_N, pso_C1, pso_C2, pso_omega, pso_f_calls_limit: PSO parameters
- error_threshold_global: Fixed absolute error threshold for outlier detection (replaces MAD)
- fator_liq: Liquidity percentage threshold for outlier detection
- ultra_low_factor: Divisor for ultra-low liquidity threshold (default: 3.0)
- max_iterations: Maximum outlier removal iterations
- min_bonds: Minimum number of bonds required
- bond_quantities: Vector with trading quantities

Returns:
- final_params: Optimal NSS parameters
- final_cost: Final cost value
- final_cash_flows: Cash flows after outlier removal
- outliers_removed: Total number of outliers removed
- iterations_used: Number of iterations used
"""
function optimize_nelson_siegel_svensson_with_mad_outlier_removal(cash_flows, ref_date, lower_bounds::Vector{Float64}, upper_bounds::Vector{Float64};
                                                                 previous_params=nothing,
                                                                 temporal_penalty_weight=0.01,
                                                                 pso_N=50, pso_C1=2.0, pso_C2=2.0,
                                                                 pso_omega=0.5, pso_f_calls_limit=1500,
                                                                 error_threshold_global=20.0, fator_liq=0.015,
                                                                 ultra_low_factor=3.0,
                                                                 max_iterations=5, min_bonds=3,
                                                                 bond_quantities=nothing,
                                                                 verbose::Bool=true)

    if isempty(cash_flows)
        return zeros(6), Inf, [], 0, 0
    end

    if bond_quantities === nothing
        bond_quantities = ones(length(cash_flows))
    end

    if verbose
        println("🎯 Iniciando otimização NSS com remoção iterativa de outliers (Threshold Fixo + Liquidez)")
        println("   Critério duplo: Erro > $(error_threshold_global) E Quantidade < $(fator_liq*100)% do total")
        println("   Títulos iniciais: $(length(cash_flows))")
    end

    current_cash_flows = copy(cash_flows)
    current_quantities = copy(bond_quantities)
    total_outliers_removed = 0

    # STEP 1: Remove ultra-low liquidity bonds FIRST (independent of error)
    # This catches data errors like volume=1 that the 2025-06-06 investigation found
    ultra_low_indices = detect_ultra_low_liquidity(current_quantities;
                                                   fator_liq=fator_liq,
                                                   ultra_low_factor=ultra_low_factor)

    if !isempty(ultra_low_indices)
        ultra_low_threshold = (fator_liq / ultra_low_factor) * sum(current_quantities)
        if verbose
            println("\n🔍 FILTRO DE LIQUIDEZ ULTRA-BAIXA (independente de erro)")
            println("   Threshold: qty < $(round(ultra_low_threshold, digits=1)) ($(round(fator_liq/ultra_low_factor*100, digits=2))% do total)")
            println("   Títulos removidos: $(length(ultra_low_indices))")
            for i in ultra_low_indices
                println("     💥 Qty=$(round(current_quantities[i], digits=0)) | Preço=R\$$(round(current_cash_flows[i][1], digits=2))")
            end
        end

        current_cash_flows, current_quantities = remove_outliers(current_cash_flows, current_quantities, ultra_low_indices)
        total_outliers_removed += length(ultra_low_indices)

        if verbose
            println("   Títulos restantes: $(length(current_cash_flows))")
        end
    end

    if length(current_cash_flows) < min_bonds
        if verbose
            println("⚠️  Menos de $min_bonds títulos após filtro de liquidez. Parando.")
        end
        return zeros(6), Inf, current_cash_flows, total_outliers_removed, 0
    end
    
    # --- Iterative Outlier Removal with Dual Criteria ---
    for iteration in 1:max_iterations
        if verbose; println("\n--- ITERAÇÃO $iteration ---"); end
        
        if length(current_cash_flows) < min_bonds
            if verbose; println("⚠️  Menos de $min_bonds títulos restantes. Parando."); end
            break
        end
        
        # Preliminary PSO fit
        pso_particles = max(pso_N ÷ 2, 20)
        pso_calls = min(pso_f_calls_limit ÷ 2, 1000)
        if verbose; println("🔄 Fit preliminar: N=$pso_particles, calls=$pso_calls"); end

        params, cost = optimize_pso_nss(current_cash_flows, ref_date, lower_bounds, upper_bounds;
                                       previous_params=previous_params,
                                       temporal_penalty_weight=temporal_penalty_weight,
                                       pso_N=pso_particles, pso_C1=pso_C1, pso_C2=pso_C2,
                                       pso_omega=pso_omega, pso_f_calls_limit=pso_calls,
                                       verbose=verbose)
        
        if verbose; println("Custo iteração $iteration: $(round(cost, digits=6))"); end

        # STEP 2: Detect outliers using FIXED threshold (not MAD-based)
        outlier_indices, errors, mean_error = detect_outliers_fixed_threshold(
            current_cash_flows, current_quantities, ref_date, params;
            error_threshold_global=error_threshold_global, fator_liq=fator_liq)

        # Print outlier summary
        total_quantity = sum(current_quantities)
        liquidity_threshold = fator_liq * total_quantity
        if verbose
            println("Erro médio = $(round(mean_error, digits=3)), Threshold fixo = $(error_threshold_global), Liquidez threshold = $(round(liquidity_threshold, digits=1))")
        end
        
        if isempty(outlier_indices)
            if verbose; println("✅ Nenhum outlier detectado. Processo concluído."); end
            break
        end
        
        # Print outlier details
        if !isempty(outlier_indices)
            if verbose
                println("⚠️  Outliers detectados: $(length(outlier_indices)) títulos")
                for i in outlier_indices
                    market_price = current_cash_flows[i][1]
                    quantity = current_quantities[i]
                    error_val = errors[i]
                    println("    💥 Erro=R\$$(round(error_val, digits=2)) | Qtde=$(round(quantity, digits=0)) | Preço=R\$$(round(market_price, digits=2))")
                end
                println("🗑️  Removidos $(length(outlier_indices)) outliers. Títulos restantes: $(length(current_cash_flows) - length(outlier_indices))")
            end
        end
        
        current_cash_flows, current_quantities = remove_outliers(current_cash_flows, current_quantities, outlier_indices)
        total_outliers_removed += length(outlier_indices)
    end
    
    if verbose; println("\n🚀 FIT FINAL INTENSIVO com $(length(current_cash_flows)) títulos limpos"); end

    final_params, final_cost = optimize_pso_nss(current_cash_flows, ref_date, lower_bounds, upper_bounds;
                                               previous_params=previous_params,
                                               temporal_penalty_weight=temporal_penalty_weight,
                                               pso_N=pso_N, pso_C1=pso_C1, pso_C2=pso_C2,
                                               pso_omega=pso_omega, pso_f_calls_limit=pso_f_calls_limit,
                                               verbose=verbose)

    if verbose
        println("\n📊 RESUMO FINAL:")
        println("   Total de outliers removidos: $total_outliers_removed")
        println("   Títulos no fit final: $(length(current_cash_flows))")
        println("   Iterações usadas: $max_iterations")
    end

    return final_params, final_cost, current_cash_flows, total_outliers_removed, max_iterations
end

"""
    calculate_out_of_sample_cost_reais(cash_flows, bond_quantities, ref_date, params; use_precalc=true)

Calculates the liquidity-weighted absolute pricing error for a set of bonds.
This serves as a meaningful out-of-sample cost metric in currency units.

- Error metric: `sum(liquidity_i * |theoretical_price_i - market_price_i|)`
- This prevents positive and negative errors from canceling each other out.

# Args
- cash_flows: A vector of (market_price, cash_flow) tuples.
- bond_quantities: A vector of corresponding trading quantities (liquidity).
- ref_date: The reference date for pricing.
- params: The NSS parameters [β0, β1, β2, β3, τ1, τ2].
- use_precalc: Whether to use pre-calculated time fractions for performance.

# Returns
- total_abs_error_reais: The total liquidity-weighted absolute error in currency units.
"""
function calculate_out_of_sample_cost_reais(cash_flows, bond_quantities, ref_date, params; use_precalc=true)
    optimized_cash_flows = use_precalc ? precompute_cash_flow_times(cash_flows, ref_date) : cash_flows
    
    total_abs_error_reais = 0.0
    
    for (i, (market_price, cash_flow)) in enumerate(optimized_cash_flows)
        theoretical_price = use_precalc ? price_bond_precalc(cash_flow, params) : price_bond(cash_flow, ref_date, params)
        
        # Use absolute error to prevent positive and negative errors from canceling out
        error_abs = abs(theoretical_price - market_price)
        
        liquidity = bond_quantities[i]
        
        # Contribution in currency units: liquidity * absolute_error
        total_abs_error_reais += liquidity * error_abs
    end
    
    return total_abs_error_reais
end

"""
    precompute_cash_flow_times(cash_flows, ref_date)

Converte cash flows do formato (market_price, [(Date, Value), ...]) 
para (market_price, [(Float64, Value), ...]) com tempos pré-calculados

# Args
- cash_flows: cash flows no formato original com datas
- ref_date: data de referência para cálculo dos tempos

# Returns
- optimized_cash_flows: cash flows com tempos pré-calculados em anos
"""
function precompute_cash_flow_times(cash_flows, ref_date)
    optimized_cash_flows = []

    for (market_price, cash_flow) in cash_flows
        cash_flow_with_times = [(yearfrac(ref_date, date), amount) for (date, amount) in cash_flow]
        push!(optimized_cash_flows, (market_price, cash_flow_with_times))
    end

    return optimized_cash_flows
end

"""
    normalize_cost_by_volume(dates::Vector{Date}, costs::Vector{Float64}) -> Float64

Calculate volume-normalized cost for a period.

This function eliminates code duplication by centralizing the logic for:
1. Loading bond data for each date
2. Calculating total trading volume
3. Normalizing costs by volume

# Arguments
- `dates`: Vector of dates to process
- `costs`: Vector of costs corresponding to each date

# Returns
- Normalized cost (total cost / total volume), or total cost if volume is zero

# Example
```julia
train_dates = [Date(2024, 1, 2), Date(2024, 1, 3)]
costs = [1000.0, 1500.0]
normalized = normalize_cost_by_volume(train_dates, costs)
```
"""
function normalize_cost_by_volume(dates::Vector{Date}, costs::Vector{Float64})
    # Sum absolute costs (prevents positive and negative from canceling)
    total_cost_reais = sum(abs.(costs))

    # Calculate total trading volume for the period
    total_volume = 0.0
    for date in dates
        try
            df = load_bacen_data(date, date)
            if nrow(df) >= 3
                _, bond_quantities, _ = generate_cash_flows_with_quantity(df, date)
                total_volume += sum(bond_quantities)
            end
        catch
            # Skip dates with data loading errors
            continue
        end
    end

    # Normalize by volume (cost per unit of volume)
    return total_volume > 0 ? total_cost_reais / total_volume : total_cost_reais
end