"""
estimation.jl - Optimization and estimation functionality

This module contains the core optimization algorithms for Nelson-Siegel-Svensson
yield curve estimation, including PSO and Levenberg-Marquardt methods.
"""

using Random, LsqFit, HTTP, JSON, LinearAlgebra, Statistics

# --- Includes ---
include("financial_math.jl")

# --- Funções de Otimização e Estimação ---

"""
    remove_outliers(cash_flows, bond_quantities, outlier_indices) -> (Vector, Vector)

Remove outliers from cash flows and bond quantities arrays.
"""
function remove_outliers(cash_flows, bond_quantities, outlier_indices)
    if isempty(outlier_indices)
        return cash_flows, bond_quantities
    end
    
    # Create mask for non-outliers
    n = length(cash_flows)
    keep_mask = trues(n)
    keep_mask[outlier_indices] .= false
    
    # Filter arrays
    filtered_cash_flows = cash_flows[keep_mask]
    filtered_quantities = bond_quantities[keep_mask]
    
    return filtered_cash_flows, filtered_quantities
end


"""
    detect_outliers_mad_and_liquidity(cash_flows, bond_quantities, ref_date, params; 
                                     fator_erro=3.0, fator_liq=0.1) -> (Vector{Int}, Vector{Float64}, Float64)

Detect outliers based on TWO simultaneous conditions:
1. Error > fator_erro × MAD  
2. Trading quantity < fator_liq % of total
"""
function detect_outliers_mad_and_liquidity(cash_flows, bond_quantities, ref_date, params; 
                                          fator_erro=3.0, fator_liq=0.1)
    if isempty(cash_flows)
        return Int[], Float64[], 0.0
    end
    
    # Calculate pricing errors for all bonds
    errors = Float64[]
    for (market_price, cash_flow) in cash_flows
        theoretical_price = price_bond(cash_flow, ref_date, params)
        error_abs = abs(theoretical_price - market_price)
        push!(errors, error_abs)
    end
    
    # Calculate MAD of errors
    mad_value = calculate_mad(errors)
    error_threshold = fator_erro * mad_value
    
    # Calculate liquidity threshold
    total_quantity = sum(bond_quantities)
    liquidity_threshold = fator_liq * total_quantity
    
    # Identify outliers that satisfy BOTH conditions
    outlier_indices = Int[]
    for (i, error) in enumerate(errors)
        quantity = bond_quantities[i]
        
        # Condition 1: High error
        high_error = error > error_threshold
        
        # Condition 2: Low liquidity
        low_liquidity = quantity < liquidity_threshold
        
        # Outlier only if BOTH conditions are true
        if high_error && low_liquidity
            push!(outlier_indices, i)
        end
    end
    
    return outlier_indices, errors, mad_value
end

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
        
        # Pricing error - usando EXATAMENTE a mesma metodologia do LM
        total_cost = 0.0
        total_weight = 0.0
        
        for (market_price, cf_with_times) in cash_flows_with_times
            if isempty(cf_with_times)
                continue
            end

            # Pre-calculated theoretical price
            theoretical_price = sum(amount / (1 + nss_rate(t, params))^t for (t, amount) in cf_with_times if t > 0)

            # Handle potential invalid price calculation
            if isnan(theoretical_price) || isinf(theoretical_price) || theoretical_price <= 0.0
                return 1e12 # Add a large penalty and continue
            end

            # Duration-based weighting (pre-calculated)
            duration = sum(t * amount / (1 + nss_rate(t, params))^t for (t, amount) in cf_with_times if t > 0) / theoretical_price
            weight = 1.0 / sqrt(max(duration, 0.1))  # Peso ANBIMA suave (raiz quadrada)
            
            # Erro absoluto em preço quadrático ponderado (igual ao LM)
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
        println("🔧 Aplicando refinamento Levenberg-Marquardt com função de custo unificada...")
    end
    
    selic_rate = get_selic_rate(ref_date; verbose=verbose)
    temporal_vertices = [0.5, 1.0, 3.0, 5.0, 10.0]

    function residuals_function(params)
        residuals = Float64[]
        total_weight = 0.0

        # Pricing residuals
        for (market_price, cash_flow) in cash_flows
            theoretical_price = price_bond(cash_flow, ref_date, params)
            duration = calculate_duration(cash_flow, ref_date, params)
            weight = 1.0 / sqrt(max(duration, 0.1))
            total_weight += weight
            
            weighted_residual = sqrt(weight) * (theoretical_price - market_price)
            push!(residuals, weighted_residual)
        end
        
        # Normalize bond residuals
        if total_weight > 0
            for i in 1:length(cash_flows)
                residuals[i] /= sqrt(total_weight)
            end
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
        # IMPLEMENTAÇÃO MANUAL DO LEVENBERG-MARQUARDT (igual ao código legado)
        params_lm = copy(pso_params)
        λ = 0.001  # Parâmetro de regularização inicial
        converged = false
        iterations = 0
        
        current_residuals = residuals_function(params_lm)
        current_cost = 0.5 * sum(current_residuals.^2)
        
        for iter in 1:max_iterations
            iterations = iter
            
            # Calcula Jacobiano numericamente
            n_params = length(params_lm)
            n_residuals = length(current_residuals)
            J = zeros(n_residuals, n_params)
            
            h = 1e-7  # Step size para diferenças finitas
            for j in 1:n_params
                params_plus = copy(params_lm)
                params_plus[j] += h
                
                # Aplica bounds manualmente
                params_plus[j] = max(min(params_plus[j], [0.25, 0.15, 0.30, 0.50, 5.0, 1.0][j]), [0.05, -0.15, -0.15, -0.15, 0.5, 0.05][j])
                
                residuals_plus = residuals_function(params_plus)
                J[:, j] = (residuals_plus - current_residuals) / h
            end
            
            # Algoritmo de Levenberg-Marquardt
            JtJ = J' * J
            JtR = J' * current_residuals
            
            # Adiciona regularização de Levenberg-Marquardt
            identity_matrix = Matrix{Float64}(LinearAlgebra.I, n_params, n_params)
            A = JtJ + λ * identity_matrix
            
            # Resolve sistema linear: A * δ = -JtR
            try
                δ = -A \ JtR
                
                # Nova tentativa de parâmetros
                params_new = params_lm + δ
                
                # Aplica bounds
                for i in 1:length(params_new)
                    lower = [-1.0, -2.5, -3.0, -5.0, 0.005, 0.005][i]
                    upper = [3.0, 2.5, 3.0, 4.0, 75.0, 60.0][i]
                    params_new[i] = max(min(params_new[i], upper), lower)
                end
                
                # Avalia novo custo
                new_residuals = residuals_function(params_new)
                new_cost = 0.5 * sum(new_residuals.^2)
                
                # Critério de aceitação
                if new_cost < current_cost
                    # Aceita passo
                    params_lm = params_new
                    current_residuals = new_residuals
                    current_cost = new_cost
                    λ = λ / 10  # Reduz regularização
                    
                    # Critério de convergência mais realista para dados financeiros
                    if norm(δ) < 1e-6 || abs(new_cost - current_cost) / max(current_cost, 1e-6) < 1e-8
                        converged = true
                        break
                    end
                else
                    # Rejeita passo
                    λ = λ * 10  # Aumenta regularização
                end
                
            catch e
                # Se sistema é singular, aumenta regularização
                λ = λ * 10
            end
            
            # Evita λ muito grande
            if λ > 1e10
                break
            end
        end
        
        # Calcula custo final usando EXATAMENTE a mesma metodologia do PSO
        total_cost = 0.0
        total_weight = 0.0
        
        # Custo dos títulos (metodologia ANBIMA)
        for (market_price, cash_flow) in cash_flows
            theoretical_price = price_bond(cash_flow, ref_date, params_lm)
            
            # Peso ANBIMA: w_i = 1/√Duration_i (versão suave)
            duration = calculate_duration(cash_flow, ref_date, params_lm)
            weight = 1.0 / sqrt(max(duration, 0.1))
            
            # Erro absoluto em preço quadrático ponderado
            error_abs = theoretical_price - market_price  
            total_cost += weight * error_abs^2
            total_weight += weight
        end
        
        # Penalidade SELIC (igual ao PSO)
        t_1day = 1/252
        r_1day_model = nss_rate(t_1day, params_lm)
        selic_penalty = 1000000.0 * (r_1day_model - selic_rate)^2
        
        # Penalização temporal (igual ao PSO)  
        temporal_penalty = 0.0
        if previous_params !== nothing
            for t in temporal_vertices
                rate_current = nss_rate(t, params_lm)
                rate_previous = nss_rate(t, previous_params)
                temporal_penalty += (rate_current - rate_previous)^2
            end
            temporal_penalty *= temporal_penalty_weight
        end
        
        # Combina TODOS os componentes (igual ao PSO)
        cost_bonds = total_cost / total_weight
        cost_lm = cost_bonds + selic_penalty + temporal_penalty

        if verbose
            println("   Convergiu: $(converged ? "✅" : "❌") em $iterations iterações")
            println("   Custo LM (unificado): $(round(cost_lm, digits=6))")
        end
        
        return params_lm, cost_lm, converged
        
    catch e
        @warn "LM optimization failed: $e. Retornando parâmetros PSO."
        
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
    
    # Initialize quantities if not provided
    if bond_quantities === nothing
        bond_quantities = ones(length(cash_flows))
    end
    
    if verbose
        println("🎯 Iniciando otimização NSS com remoção iterativa de outliers (MAD + Liquidez)")
        println("   Critério duplo: Erro > $(fator_erro) × MAD E Quantidade < $(fator_liq*100)% do total")
        println("   Títulos iniciais: $(length(cash_flows))")
    end
    
    current_cash_flows = copy(cash_flows)
    current_quantities = copy(bond_quantities)
    total_outliers_removed = 0
    
    for iteration in 1:max_iterations
        if verbose
            println()
            println("--- ITERAÇÃO $iteration ---")
        end
        
        # Check minimum bonds
        if length(current_cash_flows) < min_bonds
            if verbose
                println("⚠️  Menos de $min_bonds títulos restantes. Parando.")
            end
            break
        end
        
        # Preliminary PSO fit
        pso_calls = min(pso_f_calls_limit ÷ 2, 1000)  # Use fewer calls for preliminary fit
        pso_particles = max(pso_N ÷ 2, 20)
        
        if verbose
            println("🔄 Fit preliminar: N=$pso_particles, calls=$pso_calls")
        end
        
        if previous_params !== nothing
            if verbose
                selic_rate_pct = round(get_selic_rate(ref_date; verbose=verbose)*100, digits=2)
                println("Implementando penalidade para forçar r(1 dia) = $(selic_rate_pct)% (SELIC oficial)")
                println("Usando parâmetros do dia anterior como ponto inicial")
                println("Penalização temporal: peso = $temporal_penalty_weight")
            end
        else
            if verbose
                selic_rate_pct = round(get_selic_rate(ref_date; verbose=verbose)*100, digits=2)
                println("Implementando penalidade para forçar r(1 dia) = $(selic_rate_pct)% (SELIC oficial)")
            end
        end
        
        if verbose
            println("Hiperparâmetros PSO: N=$pso_particles, C1=$pso_C1, C2=$pso_C2, ω=$pso_omega, calls=$pso_calls")
            println("🚀 Usando tempos pré-calculados para otimização de desempenho")
            println("Executando otimização...")
        end
        
        params, cost = optimize_pso_nss(current_cash_flows, ref_date;
                                       previous_params=previous_params,
                                       temporal_penalty_weight=temporal_penalty_weight,
                                       pso_N=pso_particles, pso_C1=pso_C1, pso_C2=pso_C2,
                                       pso_omega=pso_omega, pso_f_calls_limit=pso_calls,
                                       verbose=verbose)
        
        if verbose
            println("Custo final: $(round(cost, digits=6))")
            println("Custo iteração $iteration: $(round(cost, digits=6))")
        end
        
        # Detect outliers
        outlier_indices, errors, mad_value = detect_outliers_mad_and_liquidity(
            current_cash_flows, current_quantities, ref_date, params;
            fator_erro=fator_erro, fator_liq=fator_liq)
        
        # Print outlier summary
        error_threshold = fator_erro * mad_value
        liquidity_threshold = fator_liq * sum(current_quantities)
        if verbose
            println("MAD = $(round(mad_value, digits=3)), Erro threshold = $(round(error_threshold, digits=2)), Liquidez threshold = $(round(liquidity_threshold, digits=1))")
        end
        
        if isempty(outlier_indices)
            if verbose
                println("✅ Nenhum outlier detectado. Processo concluído.")
            end
            
            # Final intensive fit
            if verbose
                println()
                println("🚀 FIT FINAL INTENSIVO com $(length(current_cash_flows)) títulos limpos")
            end
            if previous_params !== nothing
                if verbose
                    selic_rate_pct = round(get_selic_rate(ref_date; verbose=verbose)*100, digits=2)
                    println("Implementando penalidade para forçar r(1 dia) = $(selic_rate_pct)% (SELIC oficial)")
                    println("Usando parâmetros do dia anterior como ponto inicial")
                    println("Penalização temporal: peso = $temporal_penalty_weight")
                end
            else
                if verbose
                    selic_rate_pct = round(get_selic_rate(ref_date; verbose=verbose)*100, digits=2)
                    println("Implementando penalidade para forçar r(1 dia) = $(selic_rate_pct)% (SELIC oficial)")
                end
            end
            if verbose
                println("Hiperparâmetros PSO: N=$pso_N, C1=$pso_C1, C2=$pso_C2, ω=$pso_omega, calls=$pso_f_calls_limit")
                println("🚀 Usando tempos pré-calculados para otimização de desempenho")
                println("Executando otimização...")
            end
            
            final_params, final_cost = optimize_pso_nss(current_cash_flows, ref_date;
                                                       previous_params=previous_params,
                                                       temporal_penalty_weight=temporal_penalty_weight,
                                                       pso_N=pso_N, pso_C1=pso_C1, pso_C2=pso_C2,
                                                       pso_omega=pso_omega, pso_f_calls_limit=pso_f_calls_limit,
                                                       verbose=verbose)
            
            if verbose
                println("Custo final: $(round(final_cost, digits=6))")
                println("Custo final: $(round(final_cost, digits=6))")
                println()
                println("📊 RESUMO FINAL:")
                println("   Total de outliers removidos: $total_outliers_removed")
                println("   Títulos no fit final: $(length(current_cash_flows))")
                println("   Iterações usadas: $iteration")
            end
            
            return final_params, final_cost, current_cash_flows, total_outliers_removed, iteration
        end
        
        # Print outlier details
        if !isempty(outlier_indices)
            if verbose
                println("⚠️  Outliers detectados: $(length(outlier_indices)) títulos")
            end
            for i in outlier_indices
                if i <= length(current_cash_flows) && i <= length(current_quantities)
                    market_price = current_cash_flows[i][1]
                    quantity = current_quantities[i]
                    error = errors[i]
                    if verbose
                        println("    💥 Erro=R\$$(round(error, digits=2)) | Qtde=$(round(quantity, digits=0)) | Preço=R\$$(round(market_price, digits=2))")
                    end
                end
            end
            if verbose
                println("🗑️  Removidos $(length(outlier_indices)) outliers. Títulos restantes: $(length(current_cash_flows) - length(outlier_indices))")
            end
        end
        
        # Remove outliers
        current_cash_flows, current_quantities = remove_outliers(current_cash_flows, current_quantities, outlier_indices)
        total_outliers_removed += length(outlier_indices)
    end
    
    # If we reach here, we've used all iterations
    if verbose
        println()
        println("⚠️  Máximo de iterações ($max_iterations) atingido")
    end
    
    # Final fit with remaining bonds
    final_params, final_cost = optimize_pso_nss(current_cash_flows, ref_date;
                                               previous_params=previous_params,
                                               temporal_penalty_weight=temporal_penalty_weight,
                                               pso_N=pso_N, pso_C1=pso_C1, pso_C2=pso_C2,
                                               pso_omega=pso_omega, pso_f_calls_limit=pso_f_calls_limit,
                                               verbose=verbose)
    
    if verbose
        println("📊 RESUMO FINAL:")
        println("   Total de outliers removidos: $total_outliers_removed")
        println("   Títulos no fit final: $(length(current_cash_flows))")
        println("   Iterações usadas: $max_iterations")
    end
    
    return final_params, final_cost, current_cash_flows, total_outliers_removed, max_iterations
end

"""
    calculate_out_of_sample_cost_reais(cash_flows, bond_quantities, ref_date, params; use_precalc=true)

Calcula custo out-of-sample como somatório de (liquidez × erro) em reais
- Erro SEM módulo: preço_teórico - preço_mercado
- Liquidez: quantidade negociada do título
- Resultado em REAIS de erro de precificação do dia

# Args
- cash_flows: fluxos dos títulos (market_price, cash_flow)
- bond_quantities: vetor com quantidades negociadas
- ref_date: data de referência
- params: parâmetros NSS [β0, β1, β2, β3, τ1, τ2]
- use_precalc: usar tempos pré-calculados para performance

# Returns
- cost_reais: custo total em reais para o dia
"""
function calculate_out_of_sample_cost_reais(cash_flows, bond_quantities, ref_date, params; use_precalc=true)
    # Pre-calcula tempos se solicitado
    optimized_cash_flows = use_precalc ? precompute_cash_flow_times(cash_flows, ref_date) : cash_flows
    
    total_cost_reais = 0.0
    
    for (i, (market_price, cash_flow)) in enumerate(optimized_cash_flows)
        if use_precalc
            theoretical_price = price_bond_precalc(cash_flow, params)
        else
            theoretical_price = price_bond(cash_flow, ref_date, params)
        end
        
        # Erro SEM módulo (pode ser positivo ou negativo)
        error = theoretical_price - market_price
        
        # Liquidez do título
        liquidity = bond_quantities[i]
        
        # Contribuição em reais: liquidez × erro
        total_cost_reais += liquidity * error
    end
    
    return total_cost_reais
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