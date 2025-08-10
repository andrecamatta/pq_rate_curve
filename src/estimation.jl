"""
estimation.jl - Optimization and estimation functionality

This module contains the core optimization algorithms for Nelson-Siegel-Svensson
yield curve estimation, including PSO and Levenberg-Marquardt methods.
"""

using Random, LsqFit, HTTP, JSON, LinearAlgebra, Statistics

# --- Includes ---
include("financial_math.jl")
include("outlier_detection.jl") # Added to access outlier functions

# --- Funções de Otimização e Estimação ---

# Note: The functions remove_outliers and the original detect_outliers_mad_and_liquidity
# have been removed from this file to eliminate duplication.
# They are now centralized in outlier_detection.jl.

# Global cache for SELIC rates
const SELIC_CACHE = Dict{Date, Float64}()

"""
    get_selic_rate(date::Date) -> Float64

Get SELIC rate for a specific date, using cache for performance.

Parameters:
- date: Date for SELIC rate lookup

Returns:
- SELIC rate as decimal (e.g., 0.1275 for 12.75%)
"""
function get_selic_rate(date::Date; max_fallback_days=10, verbose::Bool=true)
    # 1. Checa o cache primeiro para a data original
    if haskey(SELIC_CACHE, date)
        # Força a impressão no log para monitoramento
        if verbose
            @info "Cache hit for SELIC on $(date): $(round(SELIC_CACHE[date]*100, digits=2))% a.a."
        end
        return SELIC_CACHE[date]
    end

    # 2. Loop de fallback para buscar a taxa mais recente
    for i in 0:max_fallback_days
        current_date = date - Day(i)
        
        # Otimização: se a data de fallback já está no cache, usa e encerra
        if haskey(SELIC_CACHE, current_date)
            rate = SELIC_CACHE[current_date]
            if verbose
                @info "Fallback cache hit on $(current_date) for original date $(date). Rate: $(round(rate*100, digits=2))% a.a."
            end
            SELIC_CACHE[date] = rate # Cache para a data original
            return rate
        end

        date_str = Dates.format(current_date, "dd/mm/yyyy")
        url = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.432/dados?formato=json&dataInicial=$(date_str)&dataFinal=$(date_str)"
        
        try
            if verbose
                @info "Fetching SELIC (series 432) from BACEN for $(current_date)..."
            end
            response = HTTP.get(url; readtimeout=10, retries=2)
            
            # Verifica o tipo de conteúdo para evitar erros de JSON
            content_type = HTTP.header(response, "Content-Type", "")
            if !contains(content_type, "application/json")
                body_preview = String(response.body)[1:min(100, end)]
                @warn "Unexpected content type for $(current_date): $(content_type). Body: '$(body_preview)'"
                continue # Tenta o próximo dia de fallback
            end
            
            data = JSON.parse(String(response.body))
            
            if !isempty(data)
                selic_value = data[1]["valor"]
                selic_rate = parse(Float64, selic_value) / 100.0
                
                # Sucesso: armazena no cache para a data consultada E a data original
                SELIC_CACHE[current_date] = selic_rate
                SELIC_CACHE[date] = selic_rate # Cache para a data original economiza futuras buscas
                
                if verbose
                    @info "SELIC found for $(current_date): $(round(selic_rate*100, digits=2))% a.a. Cached for both $(current_date) and $(date)."
                end
                return selic_rate
            end
        catch e
            # Apenas avisa sobre o erro e continua o fallback
            @warn "Error fetching SELIC for $(current_date): $(e). Retrying with previous day."
        end
    end

    # 3. Se o loop terminar sem sucesso, usa o último valor do cache, se houver
    if !isempty(SELIC_CACHE)
        latest_date = maximum(keys(SELIC_CACHE))
        rate = SELIC_CACHE[latest_date]
        @warn "API fetch failed after $(max_fallback_days) days. Using most recent cached rate from $(latest_date): $(round(rate*100, digits=2))% a.a."
        return rate
    end
    
    # 4. Fallback final se o cache estiver vazio
    default_rate = 0.1275 # 12.75%
    @error "CRITICAL: SELIC rate fetch failed completely and cache is empty. Using default rate: $(round(default_rate*100, digits=2))% a.a."
    return default_rate
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
function optimize_pso_nss(cash_flows, ref_date; 
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
    
    # Define objective function
    function objective(params)
        # Penalidade #1: Validade dos fatores de desconto (sem arbitragem)
        if !validate_discount_factors(params)
            return 1e9
        end

        # Penalty for negative or extreme parameters
        if any(isnan, params) || any(isinf, params)
            return 1e10
        end
        
        # Parameter bounds penalty
        penalty = 0.0
        if abs(params[5]) < 0.005 || abs(params[6]) < 0.005  # tau constraints ajustados aos bounds
            penalty += 1000.0
        end
        
        # SELIC constraint: r(1 day) should be close to SELIC
        t_1day = 1/252  # 1 dia útil em anos (igual ao LM)
        model_rate_1day = nss_rate(t_1day, params)
        selic_penalty = 1000000.0 * (model_rate_1day - selic_rate)^2  # Peso máximo para aderência extrema SELIC
        
        # Temporal continuity penalty - usando vértices como no LM
        temporal_penalty = 0.0
        if previous_params !== nothing
            temporal_vertices = [0.5, 1.0, 3.0, 5.0, 10.0]  # Igual ao LM
            for t in temporal_vertices
                rate_current = nss_rate(t, params)
                rate_previous = nss_rate(t, previous_params)
                temporal_penalty += (rate_current - rate_previous)^2
            end
            temporal_penalty *= temporal_penalty_weight
        end
        
        # Pricing error - using continuous compounding
        total_cost = 0.0
        total_weight = 0.0
        
        for (market_price, cf_with_times) in cash_flows_with_times
            if isempty(cf_with_times)
                continue
            end

            # Pre-calculated theoretical price with continuous compounding
            theoretical_price = sum(amount * exp(-nss_rate(t, params) * t) for (t, amount) in cf_with_times if t > 0)

            # Handle potential invalid price calculation
            if isnan(theoretical_price) || isinf(theoretical_price) || theoretical_price <= 0.0
                return 1e12 # Add a large penalty and continue
            end

            # Duration-based weighting (pre-calculated) with continuous compounding
            if theoretical_price > 0
                duration = sum(t * amount * exp(-nss_rate(t, params) * t) for (t, amount) in cf_with_times if t > 0) / theoretical_price
            else
                duration = 0.1 # Fallback for invalid price
            end
            weight = 1.0 / sqrt(max(duration, 0.1))
            
            # Weighted squared pricing error
            error_abs = theoretical_price - market_price
            total_cost += weight * error_abs^2
            total_weight += weight
        end
        
        # Normaliza custo dos títulos (igual ao LM)
        cost_bonds = total_weight > 0 ? total_cost / total_weight : 1e12
        
        return cost_bonds + penalty + selic_penalty + temporal_penalty
    end
    
    # PSO bounds (formato decimal: 0.01 = 1%)
    # Beta0: taxa longa (1% a 30%), Beta1: slope (±25%), Beta2/Beta3: curvaturas (±30%/±20%)
    lower_bounds = [0.01, -0.25, -0.30, -0.20, 0.5, 2.0]
    upper_bounds = [0.30, 0.25, 0.30, 0.20, 20.0, 50.0]
    
    # Initialize particles
    n_params = 6
    particles = zeros(pso_N, n_params)
    velocities = zeros(pso_N, n_params)
    personal_best = zeros(pso_N, n_params)
    personal_best_scores = fill(Inf, pso_N)
    
    # Initialize with previous params if available
    if previous_params !== nothing
        particles[1, :] = previous_params + 0.01 * randn(n_params)
    end
    
    # Random initialization for other particles
    for i in (previous_params !== nothing ? 2 : 1):pso_N
        for j in 1:n_params
            particles[i, j] = lower_bounds[j] + rand() * (upper_bounds[j] - lower_bounds[j])
        end
    end
    
    # Evaluate initial particles
    for i in 1:pso_N
        score = objective(particles[i, :])
        if score < personal_best_scores[i]
            personal_best_scores[i] = score
            personal_best[i, :] = particles[i, :]
        end
    end
    
    # Find global best
    global_best_score = minimum(personal_best_scores)
    global_best_idx = argmin(personal_best_scores)
    global_best = copy(personal_best[global_best_idx, :])
    
    # PSO main loop
    evaluations = pso_N
    while evaluations < pso_f_calls_limit
        for i in 1:pso_N
            # Update velocity
            r1, r2 = rand(2)
            velocities[i, :] = pso_omega * velocities[i, :] + 
                              pso_C1 * r1 * (personal_best[i, :] - particles[i, :]) + 
                              pso_C2 * r2 * (global_best - particles[i, :])
            
            # Update position
            particles[i, :] += velocities[i, :]
            
            # Apply bounds
            for j in 1:n_params
                particles[i, j] = clamp(particles[i, j], lower_bounds[j], upper_bounds[j])
            end
            
            # Evaluate
            score = objective(particles[i, :])
            evaluations += 1
            
            # Update personal best
            if score < personal_best_scores[i]
                personal_best_scores[i] = score
                personal_best[i, :] = particles[i, :]
                
                # Update global best
                if score < global_best_score
                    global_best_score = score
                    global_best = copy(particles[i, :])
                end
            end
            
            if evaluations >= pso_f_calls_limit
                break
            end
        end
    end
    
    return global_best, global_best_score
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
function refine_nss_with_levenberg_marquardt(cash_flows, ref_date, pso_params;
                                             max_iterations=100,
                                             show_trace=false,
                                             previous_params=nothing,
                                             temporal_penalty_weight=0.01,
                                             verbose::Bool=true)

    if verbose
        println("🔧 Aplicando refinamento Levenberg-Marquardt via LsqFit.jl...")
    end
    
    selic_rate = get_selic_rate(ref_date; verbose=verbose)
    temporal_vertices = [0.5, 1.0, 3.0, 5.0, 10.0]

    # The residuals function is the core of the optimization problem.
    # LsqFit minimizes the sum of squares of the returned vector.
    function residuals_function(params)
        residuals = Float64[]

        # Return a large value if parameters are invalid, to guide the solver
        if !validate_discount_factors(params)
            return fill(1e9, length(cash_flows) + 1 + (previous_params !== nothing ? length(temporal_vertices) : 0))
        end

        # Pricing residuals (weighted)
        total_weight = 0.0
        pricing_residuals = Float64[]
        for (market_price, cash_flow) in cash_flows
            theoretical_price = price_bond(cash_flow, ref_date, params)
            duration = calculate_duration(cash_flow, ref_date, params)
            weight = 1.0 / sqrt(max(duration, 0.1))
            total_weight += weight
            
            weighted_residual = sqrt(weight) * (theoretical_price - market_price)
            push!(pricing_residuals, weighted_residual)
        end
        
        # Normalize bond residuals
        if total_weight > 0
            append!(residuals, pricing_residuals / sqrt(total_weight))
        end

        # SELIC penalty residual
        t_1day = 1/252
        r_1day_model = nss_rate(t_1day, params)
        selic_penalty_val = 1000000.0 * (r_1day_model - selic_rate)^2
        push!(residuals, sqrt(selic_penalty_val))

        # Temporal penalty residuals
        if previous_params !== nothing
            for t in temporal_vertices
                rate_current = nss_rate(t, params)
                rate_previous = nss_rate(t, previous_params)
                temporal_penalty = temporal_penalty_weight * (rate_current - rate_previous)^2
                push!(residuals, sqrt(temporal_penalty))
            end
        end
        
        return residuals
    end

    try
        # Use LsqFit.jl for a robust, standard Levenberg-Marquardt implementation
        fit_result = LsqFit.lmfit(residuals_function, pso_params, maxIter=max_iterations, show_trace=show_trace)
        
        params_lm = fit_result.param
        converged = LsqFit.converged(fit_result)
        
        # --- Cost Calculation ---
        # Recalculate the final cost using the same methodology as the PSO
        # to ensure perfect comparability.
        
        # 1. Bond pricing cost
        total_cost = 0.0
        total_weight = 0.0
        for (market_price, cash_flow) in cash_flows
            theoretical_price = price_bond(cash_flow, ref_date, params_lm)
            duration = calculate_duration(cash_flow, ref_date, params_lm)
            weight = 1.0 / sqrt(max(duration, 0.1))
            
            error_sq = (theoretical_price - market_price)^2
            total_cost += weight * error_sq
            total_weight += weight
        end
        cost_bonds = total_weight > 0 ? total_cost / total_weight : 0.0

        # 2. SELIC penalty
        t_1day = 1/252
        r_1day_model = nss_rate(t_1day, params_lm)
        selic_penalty = 1000000.0 * (r_1day_model - selic_rate)^2
        
        # 3. Temporal penalty
        temporal_penalty = 0.0
        if previous_params !== nothing
            for t in temporal_vertices
                rate_current = nss_rate(t, params_lm)
                rate_previous = nss_rate(t, previous_params)
                temporal_penalty += (rate_current - rate_previous)^2
            end
            temporal_penalty *= temporal_penalty_weight
        end
        
        cost_lm = cost_bonds + selic_penalty + temporal_penalty

        if verbose
            println("   Convergiu com LsqFit.jl: $(converged ? "✅" : "❌")")
            println("   Custo LM (unificado): $(round(cost_lm, digits=6))")
        end
        
        return params_lm, cost_lm, converged
        
    catch e
        @warn "LsqFit.jl LM optimization failed: $e. Retornando parâmetros PSO."
        
        # Fallback to calculate PSO cost for comparison
        pso_residuals = residuals_function(pso_params)
        pso_cost = sum(pso_residuals.^2)
        
        return pso_params, pso_cost, false
    end
end

"""
    optimize_nelson_siegel_svensson_with_mad_outlier_removal(cash_flows, ref_date;
                                                           previous_params=nothing,
                                                           temporal_penalty_weight=0.01,
                                                           pso_N=50, pso_C1=2.0, pso_C2=2.0,
                                                           pso_omega=0.5, pso_f_calls_limit=1500,
                                                           fator_erro=3.0, fator_liq=0.1,
                                                           max_iterations=5, min_bonds=3,
                                                           bond_quantities=nothing,
                                                           use_liquidity_weights=false) -> (Vector{Float64}, Float64, Vector, Int, Int)

Main optimization function with iterative outlier removal using MAD and liquidity criteria.

Parameters:
- cash_flows: Vector of (market_price, cash_flow) tuples
- ref_date: Reference date
- previous_params: Previous day parameters for temporal continuity  
- temporal_penalty_weight: Weight for temporal penalty
- pso_N, pso_C1, pso_C2, pso_omega, pso_f_calls_limit: PSO parameters
- fator_erro: MAD multiplier for outlier detection
- fator_liq: Liquidity percentage threshold for outlier detection
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
function optimize_nelson_siegel_svensson_with_mad_outlier_removal(cash_flows, ref_date;
                                                                 previous_params=nothing,
                                                                 temporal_penalty_weight=0.01,
                                                                 pso_N=50, pso_C1=2.0, pso_C2=2.0,
                                                                 pso_omega=0.5, pso_f_calls_limit=1500,
                                                                 fator_erro=3.0, fator_liq=0.1,
                                                                 max_iterations=5, min_bonds=3,
                                                                 bond_quantities=nothing,
                                                                 verbose::Bool=true)
    
    if isempty(cash_flows)
        return zeros(6), Inf, [], 0, 0
    end
    
    if bond_quantities === nothing
        bond_quantities = ones(length(cash_flows))
    end

    # --- Stage 1: Liquidity Pre-filtering ---
    total_quantity = sum(bond_quantities)
    liquidity_threshold = fator_liq * total_quantity

    liquid_mask = [q >= liquidity_threshold for q in bond_quantities]

    initial_cash_flows = [cf for (cf, is_liquid) in zip(cash_flows, liquid_mask) if is_liquid]
    initial_quantities = [q for (q, is_liquid) in zip(bond_quantities, liquid_mask) if is_liquid]

    illiquid_removed_count = length(cash_flows) - length(initial_cash_flows)
    
    if verbose
        println("🎯 Iniciando otimização NSS com remoção de outliers em duas fases")
        println("   Fase 1: Filtro de liquidez (remove títulos com < $(round(liquidity_threshold, digits=1)) de volume)")
        println("   Removidos por iliquidez: $illiquid_removed_count")
        println("   Títulos restantes para otimização: $(length(initial_cash_flows))")
        println("   Fase 2: Remoção iterativa por erro de preço (MAD > $(fator_erro)σ)")
    end
    
    current_cash_flows = copy(initial_cash_flows)
    current_quantities = copy(initial_quantities)
    total_outliers_removed = illiquid_removed_count
    
    # --- Stage 2: Iterative Error-based Outlier Removal ---
    for iteration in 1:max_iterations
        if verbose; println("\n--- ITERAÇÃO $iteration ---"); end
        
        if length(current_cash_flows) < min_bonds
            if verbose; println("⚠️  Menos de $min_bonds títulos restantes. Parando."); end
            break
        end
        
        pso_calls = min(pso_f_calls_limit ÷ 2, 1000)
        pso_particles = max(pso_N ÷ 2, 20)
        if verbose; println("🔄 Fit preliminar: N=$pso_particles, calls=$pso_calls"); end
        
        params, cost = optimize_pso_nss(current_cash_flows, ref_date;
                                       previous_params=previous_params,
                                       temporal_penalty_weight=temporal_penalty_weight,
                                       pso_N=pso_particles, pso_C1=pso_C1, pso_C2=pso_C2,
                                       pso_omega=pso_omega, pso_f_calls_limit=pso_calls,
                                       verbose=verbose)
        
        if verbose; println("Custo iteração $iteration: $(round(cost, digits=6))"); end
        
        outlier_indices, errors, mad_value = detect_outliers_mad(
            current_cash_flows, ref_date, params;
            fator_erro=fator_erro)
        
        error_threshold = fator_erro * mad_value
        if verbose; println("MAD = $(round(mad_value, digits=3)), Erro threshold = $(round(error_threshold, digits=2))"); end
        
        if isempty(outlier_indices)
            if verbose; println("✅ Nenhum outlier de preço detectado. Processo concluído."); end
            # No outliers found, break loop and proceed to final fit
            break
        end
        
        if verbose
            println("⚠️  Outliers de preço detectados: $(length(outlier_indices)) títulos")
            for i in outlier_indices
                market_price = current_cash_flows[i][1]
                quantity = current_quantities[i]
                error_val = errors[i]
                println("    💥 Erro=R\$$(round(error_val, digits=2)) | Qtde=$(round(quantity, digits=0)) | Preço=R\$$(round(market_price, digits=2))")
            end
            println("🗑️  Removidos $(length(outlier_indices)) outliers. Títulos restantes: $(length(current_cash_flows) - length(outlier_indices))")
        end
        
        current_cash_flows, current_quantities = remove_outliers(current_cash_flows, current_quantities, outlier_indices)
        total_outliers_removed += length(outlier_indices)
    end
    
    if verbose; println("\n🚀 FIT FINAL INTENSIVO com $(length(current_cash_flows)) títulos limpos"); end
    
    final_params, final_cost = optimize_pso_nss(current_cash_flows, ref_date;
                                               previous_params=previous_params,
                                               temporal_penalty_weight=temporal_penalty_weight,
                                               pso_N=pso_N, pso_C1=pso_C1, pso_C2=pso_C2,
                                               pso_omega=pso_omega, pso_f_calls_limit=pso_f_calls_limit,
                                               verbose=verbose)
    
    if verbose
        println("\n📊 RESUMO FINAL:")
        println("   Total de outliers removidos: $total_outliers_removed ($illiquid_removed_count por liquidez)")
        println("   Títulos no fit final: $(length(current_cash_flows))")
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