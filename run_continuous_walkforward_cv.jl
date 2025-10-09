#!/usr/bin/env julia

using Distributed, TOML, Logging

# Configuração de workers para processamento paralelo a partir do config.toml
function setup_workers()
    config_service = PQRateCurve.default_config()
    cv_config = PQRateCurve.get_cv_config(config_service)
    max_cores = get(cv_config, "max_cores", Sys.CPU_THREADS)

    if nworkers() == 1
        n_cores = min(Sys.CPU_THREADS, max_cores)
        println("🚀 Adicionando $n_cores workers para processamento paralelo...")
        addprocs(n_cores)
    end
end

setup_workers()

# Carrega o módulo no processo principal e em todos os workers
using PQRateCurve
@everywhere using PQRateCurve

# 3. Carrega dependências básicas em todos os processos
using Dates, Statistics, DataFrames, TOML
@everywhere using Dates, Statistics, DataFrames, TOML, CSV, HTTP, ZipFile, LinearAlgebra, Random, Optim, JSON

# 4. Adiciona Metaheuristics para Otimização
using Metaheuristics
@everywhere using Metaheuristics


println("🎯 WALK-FORWARD CONTÍNUO PSO+L-BFGS - Blocos de 30 dias")
println("=" ^ 60)
println("🔄 Modo: PARALELO ($(nworkers()) workers)")

@everywhere struct PSOHyperparams
    N::Int
    C1::Float64
    C2::Float64
    ω::Float64
    f_calls_limit::Int
    use_lbfgs::Bool
    temporal_penalty_weight::Float64
    error_threshold_global::Float64  # Replaced MAD-based threshold
    fator_liq::Float64
    ultra_low_factor::Float64  # NEW: independent ultra-low liquidity filter
end

# Cache global para armazenar resultados de todas as avaliações da Otimização Bayesiana
global BAYESIAN_RESULTS = Vector{Any}()
global BAYESIAN_COUNTER = 0
global BAYESIAN_START_TIME = 0.0



# Carrega blocos de treino/teste a partir do config.toml
function get_continuous_blocks_from_config()
    config_service = PQRateCurve.default_config()
    cv_config = PQRateCurve.get_cv_config(config_service)
    block_configs = get(cv_config, "blocks", [])

    if isempty(block_configs)
        error("Nenhum bloco de cross-validation definido em config.toml")
    end

    blocks = [(
        train_start=Date(b["train_start"]),
        train_end=Date(b["train_end"]),
        test_start=Date(b["test_start"]),
        test_end=Date(b["test_end"])
    ) for b in block_configs]
    
    return blocks
end


# Treina modelo sequencialmente aproveitando previous_params
@everywhere function train_sequential(pso_params::PSOHyperparams, train_dates::Vector{Date}, train_max_iterations::Int, lbfgs_max_iterations_cv::Int, verbose::Bool = true)
    best_params = nothing
    costs = Float64[]
    successful_days = 0

    # Load config file inside the worker using ConfigService
    config_service = PQRateCurve.load_config("config.toml")
    config = PQRateCurve.get_raw_config(config_service)

    for (day_idx, train_date) in enumerate(train_dates)
        try
            df = load_bacen_data(train_date, train_date)

            if nrow(df) < 3
                if verbose
                    println("❌ Dados insuficientes para $train_date: $(nrow(df)) linhas")
                end
                continue
            end

            # Aproveita parâmetros do dia anterior
            previous_params = (day_idx > 1 && best_params !== nothing) ? best_params : nothing

            # Gera cash flows com informação de quantidade para MAD híbrido
            cash_flows, bond_quantities, _ = generate_cash_flows_with_quantity(df, train_date)

            # Get bounds from config (single source of truth)
            lower_bounds, upper_bounds = get_pso_bounds(config)

            # Uses new fixed threshold + ultra-low liquidity filter
            params, cost, final_cash_flows, _, _ = fit_nss(
                cash_flows, train_date, lower_bounds, upper_bounds;
                previous_params=previous_params,
                temporal_penalty_weight=pso_params.temporal_penalty_weight,
                pso_N=pso_params.N,
                pso_C1=pso_params.C1,
                pso_C2=pso_params.C2,
                pso_omega=pso_params.ω,
                pso_f_calls_limit=pso_params.f_calls_limit,
                error_threshold_global=pso_params.error_threshold_global,
                fator_liq=pso_params.fator_liq,
                ultra_low_factor=pso_params.ultra_low_factor,
                max_iterations=train_max_iterations,
                bond_quantities=bond_quantities,
                verbose=verbose
            )
            
            # Aplica L-BFGS se solicitado
            if pso_params.use_lbfgs
                try
                    params_lbfgs, cost_lbfgs, lbfgs_success = refine_nss_with_lbfgs(
                        final_cash_flows, train_date, params, lower_bounds, upper_bounds;
                        max_iterations=lbfgs_max_iterations_cv, show_trace=false,
                        previous_params=previous_params,
                        temporal_penalty_weight=pso_params.temporal_penalty_weight,
                        verbose=verbose
                    )
                    
                    if lbfgs_success && cost_lbfgs < cost
                        params = params_lbfgs
                        cost = cost_lbfgs
                    end
                catch
                    # Mantém PSO se L-BFGS falhar
                end
            end
            
            best_params = params
            
            # Calcula custo out-of-sample em reais para o dia de treino
            cost_reais = calculate_out_of_sample_cost_reais(final_cash_flows, bond_quantities[1:length(final_cash_flows)], train_date, params)
            push!(costs, cost_reais)
            successful_days += 1
            
        catch e
            if verbose
                println("❌ Erro em $train_date: $e")
            end
            continue
        end
    end
    
    if successful_days > 0 && best_params !== nothing
        # Normalize costs by trading volume (DRY - uses centralized function)
        normalized_train_cost = normalize_cost_by_volume(train_dates, costs)

        return best_params, normalized_train_cost, successful_days
    else
        return nothing, Inf, 0
    end
end

# Testa modelo treinado sequencialmente no período de teste
@everywhere function test_sequential(initial_params, pso_params::PSOHyperparams, test_dates::Vector{Date}, train_max_iterations::Int, lbfgs_max_iterations_cv::Int, stability_vertices::Vector{Float64}, verbose::Bool = true)
    test_costs = Float64[]
    test_params_history = []
    successful_days = 0
    current_params = initial_params

    # Load config file inside the worker using ConfigService
    config_service = PQRateCurve.load_config("config.toml")
    config = PQRateCurve.get_raw_config(config_service)

    for (_, test_date) in enumerate(test_dates)
        try
            df = load_bacen_data(test_date, test_date)
            
            if nrow(df) < 3
                continue
            end
            
            # Re-otimiza para o dia de teste, usando o dia anterior como base
            previous_params = current_params

            # Gera cash flows com informação de quantidade para MAD híbrido
            cash_flows, bond_quantities, _ = generate_cash_flows_with_quantity(df, test_date)

            # CORREÇÃO VAZAMENTO DE DADOS: Armazena dados brutos do dia
            raw_cash_flows = copy(cash_flows)

            # Get bounds from config (single source of truth)
            lower_bounds, upper_bounds = get_pso_bounds(config)

            # Uses new fixed threshold + ultra-low liquidity filter (obtains parameters)
            params, optimization_cost, final_cash_flows, _, _ = fit_nss(
                cash_flows, test_date, lower_bounds, upper_bounds;
                previous_params=previous_params,
                temporal_penalty_weight=pso_params.temporal_penalty_weight,
                pso_N=pso_params.N,
                pso_C1=pso_params.C1,
                pso_C2=pso_params.C2,
                pso_omega=pso_params.ω,
                pso_f_calls_limit=pso_params.f_calls_limit,
                error_threshold_global=pso_params.error_threshold_global,
                fator_liq=pso_params.fator_liq,
                ultra_low_factor=pso_params.ultra_low_factor,
                max_iterations=train_max_iterations,
                bond_quantities=bond_quantities,
                verbose=verbose
            )
            
            # CORREÇÃO VAZAMENTO DE DADOS: Calcula custo out-of-sample em reais usando dados brutos completos
            # Extrai bond_quantities correspondentes aos raw_cash_flows
            raw_bond_quantities = bond_quantities[1:length(raw_cash_flows)]
            cost_reais = calculate_out_of_sample_cost_reais(raw_cash_flows, raw_bond_quantities, test_date, params)
            
            # Aplica L-BFGS se solicitado
            if pso_params.use_lbfgs
                try
                    params_lbfgs, _, lbfgs_success = refine_nss_with_lbfgs(
                        final_cash_flows, test_date, params, lower_bounds, upper_bounds;
                        max_iterations=lbfgs_max_iterations_cv, show_trace=false,
                        previous_params=previous_params,
                        temporal_penalty_weight=pso_params.temporal_penalty_weight,
                        verbose=verbose
                    )
                    
                    # CORREÇÃO VAZAMENTO DE DADOS: Avalia L-BFGS usando dados brutos completos em reais
                    cost_lbfgs_reais = calculate_out_of_sample_cost_reais(raw_cash_flows, raw_bond_quantities, test_date, params_lbfgs)
                    
                    # Compara custos absolutos em reais 
                    if lbfgs_success && abs(cost_lbfgs_reais) < abs(cost_reais)
                        params = params_lbfgs
                        cost_reais = cost_lbfgs_reais
                    end
                catch
                    # Mantém PSO se L-BFGS falhar
                end
            end
            
            current_params = params
            push!(test_costs, cost_reais)
            push!(test_params_history, current_params)
            successful_days += 1
            
        catch e
            if verbose
                println("❌ Erro no teste sequencial em $test_date: $e. Interrompendo bloco.")
            end
            break
        end
    end
    
    if successful_days > 0
        # Normalize costs by trading volume (DRY - uses centralized function)
        # Only process dates that were successfully tested (break on error means sequential days)
        normalized_test_cost = normalize_cost_by_volume(test_dates[1:length(test_costs)], test_costs)

        # Calcula estabilidade dos vértices
        vertex_stability = calculate_vertex_stability(test_params_history, stability_vertices)
        
        return (
            normalized_test_cost = normalized_test_cost,
            successful_days = successful_days,
            vertex_stability = vertex_stability
        )
    else
        return nothing
    end
end

# Calcula estabilidade dos vértices da curva
@everywhere function calculate_vertex_stability(params_history, vertices)
    if length(params_history) < 2
        return 0.0
    end
    
    total_variation = 0.0
    vertex_count = 0
    
    for vertex in vertices
        rates = [nss_rate(vertex, params) for params in params_history]
        
        if length(rates) >= 2
            # Variação quadrática média entre dias consecutivos (em pontos base)
            daily_variations = [(rates[i] - rates[i-1])^2 for i in 2:length(rates)]
            avg_variation = sqrt(mean(daily_variations)) * 10000  # pontos base
            
            total_variation += avg_variation
            vertex_count += 1
        end
    end
    
    return vertex_count > 0 ? total_variation / vertex_count : 0.0
end

# Processa um bloco individual (paralelizável)
@everywhere function process_single_block(pso_params::PSOHyperparams, block, block_idx, cv_config::Dict, verbose::Bool = true)
    worker_id = myid()
    
    train_max_iterations = get(cv_config, "train_max_iterations", 2)
    lbfgs_max_iterations_cv = get(cv_config, "lbfgs_max_iterations_cv", 50)
    stability_vertices = get(cv_config, "stability_vertices", [0.5, 1.0, 3.0, 5.0, 10.0, 15.0])

    if verbose
        print("🔧 Worker $worker_id: [$block_idx/$(length(cv_config["blocks"]))] ")
        print("Treino: $(block.train_start) a $(block.train_end), ")
        print("Teste: $(block.test_start) a $(block.test_end) ")
    end
    
    # Gera datas úteis
    train_dates = get_business_dates(block.train_start, block.train_end)
    test_dates = get_business_dates(block.test_start, block.test_end)
    
    if isempty(train_dates) || isempty(test_dates)
        if verbose
            println("❌ Sem datas úteis")
        end
        return nothing
    end
    
    # TREINO: Sequencial com previous_params
    trained_params, train_cost, train_days = train_sequential(pso_params, train_dates, train_max_iterations, lbfgs_max_iterations_cv, verbose)
    
    if trained_params === nothing || train_days < 3
        if verbose
            println("❌ Falha no treino ($train_days dias)")
        end
        return nothing
    end
    
    # TESTE: Avalia parâmetros treinados sequencialmente
    test_result = test_sequential(trained_params, pso_params, test_dates, train_max_iterations, lbfgs_max_iterations_cv, stability_vertices, verbose)
    
    if test_result === nothing
        if verbose
            println("❌ Falha no teste")
        end
        return nothing
    end
    
    result = (
        block_idx = block_idx,
        normalized_train_cost = train_cost,
        train_days = train_days,
        normalized_test_cost = test_result.normalized_test_cost,
        test_days = test_result.successful_days,
        vertex_stability = test_result.vertex_stability,
        period = "$(block.train_start)-$(block.test_end)"
    )
    
    if verbose
        println("✅ Treino: $(round(train_cost, digits=6)) norm ($(train_days)d), Teste: $(round(test_result.normalized_test_cost, digits=6)) norm ($(test_result.successful_days)d), Estab: $(round(test_result.vertex_stability, digits=1))bp")
    end
    
    return result
end

# Walk-forward para uma configuração PSO - VERSÃO PARALELA
function continuous_walkforward_single_config(pso_params::PSOHyperparams, blocks, cv_config::Dict)
    config_name = "N=$(pso_params.N)_C1=$(round(pso_params.C1,digits=2))_LBFGS=$(pso_params.use_lbfgs)_TW=$(round(pso_params.temporal_penalty_weight,digits=4))_ERR=$(pso_params.error_threshold_global)_LIQ=$(pso_params.fator_liq)_ULF=$(pso_params.ultra_low_factor)"
    
    # Mensagem simplificada - removida a redundante
    println("   🔄 Distribuindo $(length(blocks)) blocos entre $(nworkers()) workers...")
    
    # PARALELIZAÇÃO ROBUSTA: Usa pmap em vez de @distributed para melhor tratamento de erros
    block_tasks = [(pso_params, blocks[i], i, cv_config, false) for i in 1:length(blocks)]
    results_list = pmap(args -> process_single_block(args...), block_tasks)
    results_raw = filter(x -> x !== nothing, results_list)
    
    println("   ✅ Blocos processados: $(length(results_raw))/$(length(blocks))")
    
    return (pso_params, results_raw)
end

# Helper function to calculate the simplified hybrid score for Bayesian optimization
function _calculate_bayesian_score(avg_test_cost, avg_vertex_stability, overfitting_ratio, config)
    scaling_config = get(get(config, "cross_validation", Dict()), "bayesian_objective_scaling", Dict())
    cost_scale = get(scaling_config, "cost_score", 50.0)
    stability_scale = get(scaling_config, "stability_score", 20.0)
    overfitting_penalty_mult = get(scaling_config, "overfitting_penalty", 2.0)

    cost_score = avg_test_cost / cost_scale
    stability_score = avg_vertex_stability / stability_scale
    overfitting_penalty = max(0, overfitting_ratio - 1.0) * overfitting_penalty_mult

    return cost_score + stability_score + overfitting_penalty
end

# Função objetivo para Otimização Bayesiana
function bayesian_objective(params_vector)
    global BAYESIAN_COUNTER, BAYESIAN_START_TIME
    BAYESIAN_COUNTER += 1
    
    # Proteção adicional contra execução excessiva em modo teste - apenas uma vez
    config_service = PQRateCurve.default_config()
    config = PQRateCurve.get_raw_config(config_service)
    max_configs = config["validation"]["num_hyperparameter_configs"]
    if BAYESIAN_COUNTER > max_configs * 20  # Mais permissivo para o DE funcionar
        if BAYESIAN_COUNTER == max_configs * 20 + 1  # Mostra mensagem apenas uma vez
            println("⚠️  Limite de segurança atingido - modo teste concluído após explorar $(BAYESIAN_COUNTER-1) configurações")
        end
        return 1000.0  # Retorna penalidade alta para parar
    end
    
    # Extrai parâmetros do vetor
    # [N, C1, C2, omega, f_calls, use_lbfgs_prob, temporal_penalty_weight, error_threshold_global, fator_liq, ultra_low_factor]
    N = params_vector[1]
    C1 = params_vector[2]
    C2 = params_vector[3]
    omega = params_vector[4]
    f_calls_idx = params_vector[5]
    use_lbfgs_prob = params_vector[6]
    temporal_penalty_weight = params_vector[7]
    error_threshold_global = params_vector[8]
    fator_liq = params_vector[9]
    ultra_low_factor = params_vector[10]

    # Converte probabilidade use_lbfgs para booleano - FIXADO EM FALSE
    use_lbfgs = false  # use_lbfgs_prob > 0.5

    # f_calls como valor contínuo
    f_calls = round(Int, params_vector[5])

    # Cria parâmetros PSO
    pso_params = PSOHyperparams(
        round(Int, N),
        C1, C2, omega,
        f_calls,
        use_lbfgs,
        temporal_penalty_weight,
        error_threshold_global,
        fator_liq,
        ultra_low_factor
    )
    
    # Calcula tempo estimado
    elapsed = time() - BAYESIAN_START_TIME
    avg_time_per_config = elapsed > 0 ? elapsed / (BAYESIAN_COUNTER - 1) : 0
    
    println("⚙️  [$BAYESIAN_COUNTER] Avaliando configuração: N=$(pso_params.N), C1=$(round(pso_params.C1,digits=2)), C2=$(round(pso_params.C2,digits=2)), ω=$(round(pso_params.ω,digits=2)), F=$(pso_params.f_calls_limit), L-BFGS=$(pso_params.use_lbfgs), TW=$(round(pso_params.temporal_penalty_weight,digits=4)), ERR=$(round(pso_params.error_threshold_global,digits=2)), LIQ=$(round(pso_params.fator_liq,digits=4)), ULF=$(round(pso_params.ultra_low_factor,digits=2))")
    if BAYESIAN_COUNTER > 1
        println("   ⏱️  Tempo médio por configuração: $(round(avg_time_per_config, digits=1))s")
    end
    
    # Executa walk-forward para esta configuração
    cv_config = get(config, "cross_validation", Dict())
    blocks = get_continuous_blocks_from_config()
    _, results = continuous_walkforward_single_config(pso_params, blocks, cv_config)
    
    # Calcula métricas agregadas (não normalizadas ainda)
    test_costs = [r.normalized_test_cost for r in results if r.normalized_test_cost > 0]
    train_costs = [r.normalized_train_cost for r in results if r.normalized_train_cost > 0]
    vertex_stabilities = [r.vertex_stability for r in results if r.vertex_stability > 0]
    
    if isempty(test_costs) || isempty(train_costs) || isempty(vertex_stabilities)
        # Penaliza configurações que falharam
        println("❌ Configuração falhou - retornando penalidade")
        return 1000.0  # Alto valor para minimização
    end
    
    # Métricas principais com variâncias
    avg_test_cost = mean(test_costs)
    test_cost_std = length(test_costs) > 1 ? std(test_costs) : 0.0
    
    avg_train_cost = mean(train_costs)
    
    avg_vertex_stability = mean(vertex_stabilities)
    stability_std = length(vertex_stabilities) > 1 ? std(vertex_stabilities) : 0.0
    
    overfitting_ratio = avg_test_cost / avg_train_cost
    
    simple_score = _calculate_bayesian_score(avg_test_cost, avg_vertex_stability, overfitting_ratio, config)
    
    # Armazena resultado completo para análise posterior
    result_data = Dict(
        "pso_params" => pso_params,
        "avg_test_cost" => avg_test_cost,
        "test_cost_std" => test_cost_std,
        "avg_train_cost" => avg_train_cost,
        "avg_vertex_stability" => avg_vertex_stability,
        "stability_std" => stability_std,
        "overfitting_ratio" => overfitting_ratio,
        "blocks_completed" => length(test_costs),
        "simple_score" => simple_score,
        "detailed_results" => results
    )
    
    push!(BAYESIAN_RESULTS, result_data)
    
    config_time = time() - BAYESIAN_START_TIME - (BAYESIAN_COUNTER - 1) * avg_time_per_config
    println("✅ [$BAYESIAN_COUNTER] Concluída em $(round(config_time, digits=1))s | Score: $(round(simple_score, digits=3)), Teste: $(round(avg_test_cost, digits=3)), Overfitting: $(round(overfitting_ratio, digits=3))")
    
    return simple_score  # Metaheuristics.jl minimiza esta função
end

# Execução principal com Otimização Bayesiana
function run_continuous_walkforward()
    println("⚙️  Configurando walk-forward contínuo com OTIMIZAÇÃO BAYESIANA...")
    
    # Carrega configuração usando ConfigService
    config_service = PQRateCurve.default_config()
    config = PQRateCurve.get_raw_config(config_service)
    validation_config = get(config, "validation", Dict())
    cv_config = PQRateCurve.get_cv_config(config_service)
    num_evaluations = get(validation_config, "num_hyperparameter_configs", 20)

    blocks = get_continuous_blocks_from_config()

    println("📊 Blocos contínuos: $(length(blocks))")
    println("📊 Avaliações Bayesianas: $num_evaluations")
    println("📊 Cada bloco: 30 dias treino → 30 dias teste (conforme config.toml)")
    # Get hyperparameter ranges for display
    hyperparams_config = PQRateCurve.get_hyperparams_config(config_service)
    
    println("📊 Espaço de busca (config.toml):")
    println("   • N ∈ [$(get(hyperparams_config, "N_min", 25)), $(get(hyperparams_config, "N_max", 80))] (população PSO)")
    println("   • C1 ∈ [$(get(hyperparams_config, "C1_min", 0.5)), $(get(hyperparams_config, "C1_max", 3.5))] (aceleração cognitiva)")
    println("   • C2 ∈ [$(get(hyperparams_config, "C2_min", 0.5)), $(get(hyperparams_config, "C2_max", 3.0))] (aceleração social)")
    println("   • ω ∈ [$(get(hyperparams_config, "omega_min", 0.1)), $(get(hyperparams_config, "omega_max", 0.9))] (peso de inércia)")
    println("   • f_calls ∈ [$(get(hyperparams_config, "f_calls_min", 600)), $(get(hyperparams_config, "f_calls_max", 2500))] (limite de avaliações)")
    println("   • use_lbfgs ∈ [$(get(hyperparams_config, "use_lbfgs_prob_min", 0.0)), $(get(hyperparams_config, "use_lbfgs_prob_max", 1.0))] (prob. refinamento L-BFGS)")
    println("   • temporal_penalty ∈ [$(get(hyperparams_config, "temporal_penalty_min", 0.0001)), $(get(hyperparams_config, "temporal_penalty_max", 0.2))] (penalidade temporal)")
    println("   • error_threshold_global ∈ [$(get(hyperparams_config, "error_threshold_global_min", 10.0)), $(get(hyperparams_config, "error_threshold_global_max", 50.0))] (threshold fixo de erro)")
    println("   • fator_liq ∈ [$(get(hyperparams_config, "fator_liq_min", 0.001)), $(get(hyperparams_config, "fator_liq_max", 0.015))] (fator liquidez)")
    println("   • ultra_low_factor ∈ [$(get(hyperparams_config, "ultra_low_factor_min", 2.0)), $(get(hyperparams_config, "ultra_low_factor_max", 5.0))] (filtro ultra-baixa liquidez)")
    println("📊 Regimes testados: Crise-Política-2015, Recessão-2016, Recuperação-2018, Pandemia-2020, Inflação-2022, Normalização-2024")
    println("📊 Método: Metaheuristics.jl with Differential Evolution + PARALELIZAÇÃO")
    println("⏱️  Estimativa: ~$(round(num_evaluations * 0.6, digits=1)) MINUTOS - Bayesian Optimization PARALELA PROFUNDA")
    
    # Limpa cache de resultados e inicializa contadores
    global BAYESIAN_RESULTS, BAYESIAN_COUNTER, BAYESIAN_START_TIME
    BAYESIAN_RESULTS = Vector{Any}()
    BAYESIAN_COUNTER = 0
    
    println("\n🚀 Iniciando Otimização Bayesiana...")
    start_time = time()
    BAYESIAN_START_TIME = start_time

    # Hyperparameter search ranges already loaded above via ConfigService
    # (reusing the same config_service and hyperparams_config from earlier)

    # Define search space with values from config or sensible defaults
    search_range = [
        (get(hyperparams_config, "N_min", 25.0), get(hyperparams_config, "N_max", 80.0)),
        (get(hyperparams_config, "C1_min", 0.5), get(hyperparams_config, "C1_max", 3.5)),
        (get(hyperparams_config, "C2_min", 0.5), get(hyperparams_config, "C2_max", 3.0)),
        (get(hyperparams_config, "omega_min", 0.1), get(hyperparams_config, "omega_max", 0.9)),
        (get(hyperparams_config, "f_calls_min", 600.0), get(hyperparams_config, "f_calls_max", 2500.0)),
        (get(hyperparams_config, "use_lbfgs_prob_min", 0.0), get(hyperparams_config, "use_lbfgs_prob_max", 1.0)),
        (get(hyperparams_config, "temporal_penalty_min", 0.0001), get(hyperparams_config, "temporal_penalty_max", 0.2)),
        (get(hyperparams_config, "error_threshold_global_min", 10.0), get(hyperparams_config, "error_threshold_global_max", 50.0)),
        (get(hyperparams_config, "fator_liq_min", 0.001), get(hyperparams_config, "fator_liq_max", 0.015)),
        (get(hyperparams_config, "ultra_low_factor_min", 2.0), get(hyperparams_config, "ultra_low_factor_max", 5.0))
    ]
    
    # Define bounds para Metaheuristics.jl
    bounds = Matrix{Float64}(undef, length(search_range), 2)
    for (i, (lower, upper)) in enumerate(search_range)
        bounds[i, 1] = lower
        bounds[i, 2] = upper
    end
    
    # Para teste rápido, usa configuração mais simples e limitada
    if num_evaluations <= 5
        println("🎯 Modo TESTE RÁPIDO - Limitando a $(num_evaluations) avaliações")
        population_size = min(num_evaluations, 5)
        max_evaluations = num_evaluations
        println("🧬 Configuração simplificada: População=$(population_size), Máximo=$(max_evaluations) avaliações")
    else
        # Configuração normal para validação completa
        population_size = max(10, min(50, num_evaluations ÷ 2))
        max_evaluations = num_evaluations * 2  # Permite alguma exploração extra
        println("🧬 Configurando Differential Evolution: População=$(population_size), Máximo=$(max_evaluations) avaliações")
    end
    
    println("📍 Iniciando evolução da população...")
    
    # Usa algoritmo DE com configuração adaptada
    if num_evaluations <= 5
        # Para teste rápido, usa configuração muito restrita
        result = Metaheuristics.optimize(bayesian_objective, bounds, DE(N = population_size, iterations = 1))
    else
        # Para validação completa, usa configuração normal
        result = Metaheuristics.optimize(bayesian_objective, bounds, DE(N = population_size, iterations = 1))
    end
    
    elapsed_time = time() - start_time
    println("\n🏁 Evolução concluída!")
    println("✅ Otimização Bayesiana concluída em $(round(elapsed_time/60, digits=1)) minutos!")
    println("📊 Total de configurações testadas: $(length(BAYESIAN_RESULTS))")
    println("\n🔍 Analisando resultados e selecionando melhor configuração...")
    
    if isempty(BAYESIAN_RESULTS)
        println("❌ Nenhum resultado para analisar")
        return Dict(), elapsed_time, num_evaluations
    end
    
    # Calcula score híbrido completo com normalização adequada
    final_results = Dict()
    
    # Extrai métricas de todas as configurações
    all_test_costs = [r["avg_test_cost"] for r in BAYESIAN_RESULTS if r["blocks_completed"] >= length(blocks)/2]
    all_train_costs = [r["avg_train_cost"] for r in BAYESIAN_RESULTS if r["blocks_completed"] >= length(blocks)/2]
    all_stabilities = [r["avg_vertex_stability"] for r in BAYESIAN_RESULTS if r["blocks_completed"] >= length(blocks)/2]

    # Compute full ranges for inverse‑percentile ranking
    test_min = minimum(all_test_costs)
    test_max = maximum(all_test_costs)
    stab_min = minimum(all_stabilities)
    stab_max = maximum(all_stabilities)
    
    if isempty(all_test_costs)
        println("❌ Nenhuma configuração teve sucesso suficiente")
        return Dict(), elapsed_time, num_evaluations
    end
    
    # Normalização por range completo (min-max)
    test_min = minimum(all_test_costs)
    test_max = maximum(all_test_costs)
    stab_min = minimum(all_stabilities)
    stab_max = maximum(all_stabilities)
    
    # Range para overfitting
    all_overfitting_ratios = [r["overfitting_ratio"] for r in BAYESIAN_RESULTS if r["blocks_completed"] >= length(blocks)/2]
    over_min = minimum(all_overfitting_ratios)
    over_max = maximum(all_overfitting_ratios)
    
    for result in BAYESIAN_RESULTS
        if result["blocks_completed"] >= length(blocks)/2  # Filtro de qualidade
            # ---- Weighted geometric‑mean of inverse‑percentile ranks ----
            # Inverse‑percentile rank (higher = better)
            inv_test = 1.0 - (result["avg_test_cost"] - test_min) / max(test_max - test_min, eps())
            inv_stab = 1.0 - (result["avg_vertex_stability"] - stab_min) / max(stab_max - stab_min, eps())
            inv_over = 1.0 - (result["overfitting_ratio"] - over_min) / max(over_max - over_min, eps())
    
            # Ensure positivity for log‑space computation
            inv_test = max(inv_test, eps())
            inv_stab = max(inv_stab, eps())
            inv_over = max(inv_over, eps())
    
            # Weights from config
            weights_config = get(cv_config, "hybrid_score_weights", Dict("test" => 0.5, "stability" => 0.3, "overfitting" => 0.2))
            w_test = get(weights_config, "test", 0.5)
            w_stab = get(weights_config, "stability", 0.3)
            w_over = get(weights_config, "overfitting", 0.2)
    
            # Weighted geometric mean via log‑space (avoids underflow)
            hybrid_score = exp(w_test * log(inv_test) + w_stab * log(inv_stab) + w_over * log(inv_over))
            
            final_results[result["pso_params"]] = (
                avg_test_cost_normalized = result["avg_test_cost"],
                avg_train_cost_normalized = result["avg_train_cost"],
                overfitting_ratio = result["overfitting_ratio"],
                avg_vertex_stability = result["avg_vertex_stability"],
                test_cost_std = result["test_cost_std"],  # Agora calculado corretamente
                stability_std = result["stability_std"],  # Agora calculado corretamente
                blocks_completed = result["blocks_completed"],
                regimes_tested = length(blocks),
                hybrid_score_normalized = hybrid_score
            )
        end
    end
    
    return final_results, elapsed_time, num_evaluations
end



function _print_comparison_table(pso_avg_test, pso_test_std, pso_avg_train, pso_overfitting, pso_avg_stability, pso_score, pso_blocks,
                                 lbfgs_avg_test, lbfgs_test_std, lbfgs_avg_train, lbfgs_overfitting, lbfgs_avg_stability, lbfgs_score, lbfgs_blocks)
    println("\n📊 RESULTADOS DA COMPARAÇÃO FINAL:")
    println("┌─────────────────────┬─────────────────┬─────────────────┐")
    println("│ Métrica             │ PSO Puro        │ PSO+L-BFGS          │")
    println("├─────────────────────┼─────────────────┼─────────────────┤")
    println("│ Custo Teste (norm.) │ $(rpad(round(pso_avg_test, digits=6), 15)) │ $(rpad(round(lbfgs_avg_test, digits=6), 15)) │")
    println("│ Desvio Teste        │ $(rpad(round(pso_test_std, digits=6), 15)) │ $(rpad(round(lbfgs_test_std, digits=6), 15)) │")
    println("│ Custo Treino (norm.)│ $(rpad(round(pso_avg_train, digits=6), 15)) │ $(rpad(round(lbfgs_avg_train, digits=6), 15)) │")
    println("│ Overfitting Ratio   │ $(rpad(round(pso_overfitting, digits=3), 15)) │ $(rpad(round(lbfgs_overfitting, digits=3), 15)) │")
    println("│ Estabilidade (bp)   │ $(rpad(round(pso_avg_stability, digits=1), 15)) │ $(rpad(round(lbfgs_avg_stability, digits=1), 15)) │")
    println("│ Score Híbrido       │ $(rpad(round(pso_score, digits=3), 15)) │ $(rpad(round(lbfgs_score, digits=3), 15)) │")
    println("│ Blocos Concluídos   │ $(rpad(pso_blocks, 15)) │ $(rpad(lbfgs_blocks, 15)) │")
    println("└─────────────────────┴─────────────────┴─────────────────┘")
end

# Comparação final PSO vs PSO+L-BFGS usando melhor configuração encontrada
function final_pso_vs_lm_comparison(best_pso_params::PSOHyperparams, cv_config::Dict)
    println("\n" * "=" ^ 80)
    println("🥊 COMPARAÇÃO FINAL: PSO PURO vs PSO+L-BFGS")
    println("=" ^ 80)
    println("🎯 Usando melhor configuração PSO encontrada na busca Bayesiana")
    println("⚙️  Config base: N=$(best_pso_params.N), C1=$(best_pso_params.C1), C2=$(best_pso_params.C2), ω=$(best_pso_params.ω)")
    println("🔧 Parâmetros: TW=$(best_pso_params.temporal_penalty_weight), ERR=$(best_pso_params.error_threshold_global), LIQ=$(best_pso_params.fator_liq), ULF=$(best_pso_params.ultra_low_factor)")
    
    blocks = get_continuous_blocks_from_config()
    
    # Cria duas versões: PSO puro e PSO+L-BFGS
    pso_only_params = PSOHyperparams(
        best_pso_params.N,
        best_pso_params.C1,
        best_pso_params.C2,
        best_pso_params.ω,
        best_pso_params.f_calls_limit,
        false,  # PSO puro
        best_pso_params.temporal_penalty_weight,
        best_pso_params.error_threshold_global,
        best_pso_params.fator_liq,
        best_pso_params.ultra_low_factor
    )

    pso_lbfgs_params = PSOHyperparams(
        best_pso_params.N,
        best_pso_params.C1,
        best_pso_params.C2,
        best_pso_params.ω,
        best_pso_params.f_calls_limit,
        true,  # PSO+L-BFGS
        best_pso_params.temporal_penalty_weight,
        best_pso_params.error_threshold_global,
        best_pso_params.fator_liq,
        best_pso_params.ultra_low_factor
    )
    
    println("\n🚀 Executando validação cruzada para PSO PURO...")
    _, pso_results = continuous_walkforward_single_config(pso_only_params, blocks, cv_config)
    
    println("\n🚀 Executando validação cruzada para PSO+L-BFGS...")
    _, pso_lbfgs_results = continuous_walkforward_single_config(pso_lbfgs_params, blocks, cv_config)
    
    # Calcula métricas para ambos
    pso_test_costs = [r.normalized_test_cost for r in pso_results if r.normalized_test_cost > 0]
    pso_train_costs = [r.normalized_train_cost for r in pso_results if r.normalized_train_cost > 0]
    pso_stabilities = [r.vertex_stability for r in pso_results if r.vertex_stability > 0]
    
    pso_lbfgs_test_costs = [r.normalized_test_cost for r in pso_lbfgs_results if r.normalized_test_cost > 0]
    pso_lbfgs_train_costs = [r.normalized_train_cost for r in pso_lbfgs_results if r.normalized_train_cost > 0]
    pso_lbfgs_stabilities = [r.vertex_stability for r in pso_lbfgs_results if r.vertex_stability > 0]
    
    if isempty(pso_test_costs) || isempty(pso_lbfgs_test_costs)
        println("❌ Falha na comparação - dados insuficientes")
        return nothing, nothing
    end
    
    # Métricas PSO puro
    pso_avg_test = mean(pso_test_costs)
    pso_test_std = length(pso_test_costs) > 1 ? std(pso_test_costs) : 0.0
    pso_avg_train = mean(pso_train_costs)
    pso_avg_stability = mean(pso_stabilities)
    pso_overfitting = pso_avg_test / pso_avg_train
    
    # Métricas PSO+L-BFGS
    lbfgs_avg_test = mean(pso_lbfgs_test_costs)
    lbfgs_test_std = length(pso_lbfgs_test_costs) > 1 ? std(pso_lbfgs_test_costs) : 0.0
    lbfgs_avg_train = mean(pso_lbfgs_train_costs)
    lbfgs_avg_stability = mean(pso_lbfgs_stabilities)
    lbfgs_overfitting = lbfgs_avg_test / lbfgs_avg_train
    
    # Score híbrido simplificado para comparação direta
    scaling_config = get(cv_config, "bayesian_objective_scaling", Dict())
    stability_scale = get(scaling_config, "stability_score", 20.0)
    overfitting_penalty_mult = get(scaling_config, "overfitting_penalty", 2.0)

    pso_score = pso_avg_test + (pso_avg_stability / stability_scale) + max(0, pso_overfitting - 1.0) * overfitting_penalty_mult
    lbfgs_score = lbfgs_avg_test + (lbfgs_avg_stability / stability_scale) + max(0, lbfgs_overfitting - 1.0) * overfitting_penalty_mult
    
    _print_comparison_table(pso_avg_test, pso_test_std, pso_avg_train, pso_overfitting, pso_avg_stability, pso_score, length(pso_test_costs),
                            lbfgs_avg_test, lbfgs_test_std, lbfgs_avg_train, lbfgs_overfitting, lbfgs_avg_stability, lbfgs_score, length(pso_lbfgs_test_costs))
    
    # Decisão final
    pso_wins_test = pso_avg_test < lbfgs_avg_test
    pso_wins_overfitting = pso_overfitting < lbfgs_overfitting  
    pso_wins_stability = pso_avg_stability < lbfgs_avg_stability  # Menor = melhor
    pso_wins_hybrid = pso_score < lbfgs_score  # Menor = melhor
    
    println("\n🏆 ANÁLISE COMPARATIVA:")
    println("  Custo de Teste: $(pso_wins_test ? "✅ PSO" : "✅ PSO+L-BFGS") $(pso_wins_test ? "menor" : "menor") ($(abs(round(((pso_avg_test - lbfgs_avg_test) / max(pso_avg_test, lbfgs_avg_test)) * 100, digits=2)))% diferença)")
    println("  Overfitting: $(pso_wins_overfitting ? "✅ PSO" : "✅ PSO+L-BFGS") melhor ($(round(min(pso_overfitting, lbfgs_overfitting), digits=3)) vs $(round(max(pso_overfitting, lbfgs_overfitting), digits=3)))")
    println("  Estabilidade: $(pso_wins_stability ? "✅ PSO" : "✅ PSO+L-BFGS") mais estável ($(round(min(pso_avg_stability, lbfgs_avg_stability), digits=1)) vs $(round(max(pso_avg_stability, lbfgs_avg_stability), digits=1)) bp/dia)")
    
    println("\n🎯 DECISÃO FINAL:")
    if pso_wins_hybrid
        improvement_pct = ((lbfgs_score - pso_score) / lbfgs_score) * 100
        println("  🏅 VENCEDOR: PSO PURO")
        println("  📈 PSO puro é $(round(improvement_pct, digits=1))% melhor no score híbrido")
        println("  💡 RECOMENDAÇÃO: Use PSO puro - mais simples e eficiente")
        winner_params = pso_only_params
        winner_result = (
            avg_test_cost_normalized = pso_avg_test,
            avg_train_cost_normalized = pso_avg_train,
            overfitting_ratio = pso_overfitting,
            avg_vertex_stability = pso_avg_stability,
            test_cost_std = pso_test_std,
            blocks_completed = length(pso_test_costs),
            hybrid_score_normalized = 1.0 - (pso_score / (pso_score + lbfgs_score))  # Normalizado entre 0-1
        )
    else
        improvement_pct = ((pso_score - lbfgs_score) / pso_score) * 100
        println("  🏅 VENCEDOR: PSO+L-BFGS")
        println("  📈 PSO+L-BFGS é $(round(improvement_pct, digits=1))% melhor no score híbrido")
        println("  💡 RECOMENDAÇÃO: Use PSO+L-BFGS - refinamento melhora performance")
        winner_params = pso_lbfgs_params
        winner_result = (
            avg_test_cost_normalized = lbfgs_avg_test,
            avg_train_cost_normalized = lbfgs_avg_train,
            overfitting_ratio = lbfgs_overfitting,
            avg_vertex_stability = lbfgs_avg_stability,
            test_cost_std = lbfgs_test_std,
            blocks_completed = length(pso_lbfgs_test_costs),
            hybrid_score_normalized = 1.0 - (lbfgs_score / (pso_score + lbfgs_score))  # Normalizado entre 0-1
        )
    end
    
    return winner_params, winner_result
end

function _print_bayesian_ranking_table(sorted_results)
    println("\n📊 RANKING OTIMIZAÇÃO BAYESIANA POR SCORE HÍBRIDO NORMALIZADO (média across regimes):")
    for (rank, (params, result)) in enumerate(sorted_results)
        lbfgs_icon = params.use_lbfgs ? "✅" : "❌"
        hybrid_score = result.hybrid_score_normalized

        println("  $rank. N=$(params.N), C1=$(round(params.C1,digits=2)), C2=$(round(params.C2,digits=2)), ω=$(round(params.ω,digits=2)), L-BFGS=$lbfgs_icon, TW=$(round(params.temporal_penalty_weight,digits=4)), ERR=$(round(params.error_threshold_global,digits=2)), LIQ=$(round(params.fator_liq,digits=4)), ULF=$(round(params.ultra_low_factor,digits=2))")
        println("     🏆 Score HÍBRIDO: $(round(hybrid_score, digits=3)) (quanto MAIOR melhor - 0-1)")
        println("     🎯 Teste normalizado: $(round(result.avg_test_cost_normalized, digits=6)) ± $(round(result.test_cost_std, digits=6))")
        println("     🌊 Estabilidade: $(round(result.avg_vertex_stability, digits=1)) ± $(round(result.stability_std, digits=1)) bp/dia")
        println("     📚 Treino normalizado: $(round(result.avg_train_cost_normalized, digits=6))")
        println("     📊 Overfitting: $(round(result.overfitting_ratio, digits=3)) ($(result.overfitting_ratio < 1.15 ? "✅ Bom" : result.overfitting_ratio < 1.3 ? "⚠️ Médio" : "❌ Alto"))")
        println("     📦 Regimes: $(result.blocks_completed)/$(result.regimes_tested)")
        println()
    end
end

# Análise dos resultados da busca Bayesiana (PSO puro)
function analyze_continuous_results(results, _)
    if isempty(results)
        println("❌ Nenhum resultado para analisar")
        return
    end
    
    println("\n" * "=" ^ 80)
    println("🏆 WALK-FORWARD CROSS-REGIME - RESULTADOS OTIMIZAÇÃO BAYESIANA")
    println("=" ^ 80)
    
    # Usa score híbrido já calculado com normalização por percentis
    # Score híbrido: quanto MAIOR melhor (0-1, onde 1 = melhor possível)
    
    # Ordena por score híbrido (decrescente - maior score primeiro)
    sorted_results = sort(collect(results), by=x->x[2].hybrid_score_normalized, rev=true)
    
    _print_bayesian_ranking_table(sorted_results)
    
    # Análise PSO vs PSO+L-BFGS
    pso_only = [(p, r) for (p, r) in sorted_results if !p.use_lbfgs]
    pso_lbfgs = [(p, r) for (p, r) in sorted_results if p.use_lbfgs]
    
    if !isempty(pso_only) && !isempty(pso_lbfgs)
        println("📈 COMPARAÇÃO PSO vs PSO+L-BFGS (OUT-OF-SAMPLE):")
        
        best_pso = pso_only[1]
        best_lbfgs = pso_lbfgs[1]
        
        println("  🥇 Melhor PSO puro:")
        println("     Config: N=$(best_pso[1].N), C1=$(best_pso[1].C1), TW=$(best_pso[1].temporal_penalty_weight), ERR=$(best_pso[1].error_threshold_global), LIQ=$(best_pso[1].fator_liq), ULF=$(best_pso[1].ultra_low_factor)")
        println("     Teste normalizado: $(round(best_pso[2].avg_test_cost_normalized, digits=6))")
        println("     Overfitting: $(round(best_pso[2].overfitting_ratio, digits=2))")
        println("     Estabilidade: $(round(best_pso[2].avg_vertex_stability, digits=1)) bp/dia")

        println("  🥇 Melhor PSO+L-BFGS:")
        println("     Config: N=$(best_lbfgs[1].N), C1=$(best_lbfgs[1].C1), TW=$(best_lbfgs[1].temporal_penalty_weight), ERR=$(best_lbfgs[1].error_threshold_global), LIQ=$(best_lbfgs[1].fator_liq), ULF=$(best_lbfgs[1].ultra_low_factor)")
        println("     Teste normalizado: $(round(best_lbfgs[2].avg_test_cost_normalized, digits=6))")
        println("     Overfitting: $(round(best_lbfgs[2].overfitting_ratio, digits=2))")
        println("     Estabilidade: $(round(best_lbfgs[2].avg_vertex_stability, digits=1)) bp/dia")
        
        # Comparações quantitativas normalizadas
        improvement = ((best_pso[2].avg_test_cost_normalized - best_lbfgs[2].avg_test_cost_normalized) / best_pso[2].avg_test_cost_normalized) * 100
        
        println("\n🎯 CONCLUSÃO DEFINITIVA (SCORE HÍBRIDO NORMALIZADO):")
        
        # Usa o score híbrido normalizado por range completo para a decisão final
        best_pso_hybrid = best_pso[2].hybrid_score_normalized
        best_lbfgs_hybrid = best_lbfgs[2].hybrid_score_normalized
        
        println("  📊 Performance HÍBRIDA across regimes (usando score normalizado min-max):")
        
        if best_lbfgs_hybrid > best_pso_hybrid
            # Usar abs() para evitar divisão por zero se o score for 0
            improvement = ((best_lbfgs_hybrid - best_pso_hybrid) / abs(best_pso_hybrid)) * 100
            println("  ✅ PSO+L-BFGS é $(round(improvement, digits=2))% superior no score híbrido")
            println("  ✅ RECOMENDAÇÃO: Usar sistema híbrido PSO+L-BFGS")
            println("  🎯 Melhor overfitting: $(round(best_lbfgs[2].overfitting_ratio, digits=3)) vs $(round(best_pso[2].overfitting_ratio, digits=3))")
        else
            improvement = ((best_pso_hybrid - best_lbfgs_hybrid) / abs(best_lbfgs_hybrid)) * 100
            println("  ⚪ PSO puro é $(round(improvement, digits=2))% superior no score híbrido")
            println("  ⚪ RECOMENDAÇÃO: PSO puro é suficiente")
            println("  🎯 Melhor overfitting: $(round(best_pso[2].overfitting_ratio, digits=3)) vs $(round(best_lbfgs[2].overfitting_ratio, digits=3))")
        end
        
        # Análise de estabilidade
        pso_stability = [r[2].avg_vertex_stability for r in pso_only]
        lm_stability = [r[2].avg_vertex_stability for r in pso_lbfgs]
        
        println("\n🌊 ANÁLISE DE ESTABILIDADE DOS VÉRTICES (6 vértices: 0.5, 1, 3, 5, 10, 15 anos):")
        println("  PSO puro - Média: $(round(mean(pso_stability), digits=1)) bp/dia")
        println("  PSO+L-BFGS - Média: $(round(mean(lm_stability), digits=1)) bp/dia")
        
        if mean(lm_stability) < mean(pso_stability)
            println("  ✅ PSO+L-BFGS produz curvas mais estáveis")
        else
            println("  ⚠️  PSO puro produz curvas mais estáveis")
        end
        
        # Análise de overfitting
        pso_overfit = [r[2].overfitting_ratio for r in pso_only]
        lm_overfit = [r[2].overfitting_ratio for r in pso_lbfgs]
        
        println("\n📊 ANÁLISE DE OVERFITTING:")
        println("  PSO puro - Overfitting médio: $(round(mean(pso_overfit), digits=2))")
        println("  PSO+L-BFGS - Overfitting médio: $(round(mean(lm_overfit), digits=2))")
    end
end

function main()
    println("🚀 Iniciando walk-forward contínuo...")
    config_service = PQRateCurve.default_config()
    cv_config = PQRateCurve.get_cv_config(config_service)
    results, elapsed_time, num_configs = run_continuous_walkforward()

    analyze_continuous_results(results, cv_config)
    
    # Executa comparação final PSO vs PSO+L-BFGS usando melhor configuração
    final_winner_params = nothing
    final_winner_result = nothing
    
    if !isempty(results)
        # Usa os resultados já ordenados por score híbrido (decrescente - maior score primeiro)
        sorted_results = sort(collect(results), by=x->x[2].hybrid_score_normalized, rev=true)
        best_pso_params = sorted_results[1][1]
        best_pso_result = sorted_results[1][2]
        
        println("\n🎯 EXECUTANDO COMPARAÇÃO FINAL PSO vs PSO+L-BFGS...")
        println("Usando melhor configuração PSO encontrada: N=$(best_pso_params.N), C1=$(best_pso_params.C1), etc.")
        
        final_winner_params, final_winner_result = final_pso_vs_lm_comparison(best_pso_params, cv_config)
        
        if final_winner_params === nothing
            println("⚠️  Falha na comparação final - usando melhor resultado da busca Bayesiana")
            final_winner_params = best_pso_params
            final_winner_result = best_pso_result
        end
    end

    # Salva resultados do modelo definitivo
    if final_winner_params !== nothing && final_winner_result !== nothing
        println("\n💾 Salvando configuração DEFINITIVA do modelo vencedor...")
        
        best_params = final_winner_params
        best_result = final_winner_result
        best_hybrid_score = best_result.hybrid_score_normalized
        
        # Salva configuração ótima
        optimal_config = Dict{String, Any}(
            "pso" => Dict{String, Any}(
                "N" => best_params.N,
                "C1" => best_params.C1,
                "C2" => best_params.C2,
                "omega" => best_params.ω,
                "f_calls_limit" => best_params.f_calls_limit
            ),
            "optimization" => Dict{String, Any}(
                "use_lbfgs" => best_params.use_lbfgs,
                "temporal_penalty_weight" => best_params.temporal_penalty_weight
            ),
            "outlier_detection" => Dict{String, Any}(
                "error_threshold_global" => best_params.error_threshold_global,
                "fator_liq" => best_params.fator_liq,
                "ultra_low_factor" => best_params.ultra_low_factor
            )
        )
        
        performance_metrics = Dict{String, Any}(
            "hybrid_score_normalized" => round(best_hybrid_score, digits=4),
            "avg_test_cost_normalized" => round(best_result.avg_test_cost_normalized, digits=8),
            "avg_train_cost_normalized" => round(best_result.avg_train_cost_normalized, digits=8),
            "overfitting_ratio" => round(best_result.overfitting_ratio, digits=3),
            "vertex_stability_bp_per_day" => round(best_result.avg_vertex_stability, digits=2),
            "blocks_completed" => best_result.blocks_completed,
            "regimes_tested" => 6,  # Total regimes tested
            "execution_time_minutes" => round(elapsed_time/60, digits=1),
            "methodology" => best_params.use_lbfgs ? "PSO_plus_LBFGS_final_comparison_winner" : "PSO_only_final_comparison_winner",
            "cv_method" => "cross_regime_walk_forward_30days_with_final_pso_vs_lbfgs_comparison",
            "period" => "2014-2024_all_regimes",
            "selection_method" => "bayesian_optimization_plus_final_head_to_head_comparison"
        )
        
        # Chama a função usando TOML diretamente para salvar
        output_data = Dict{String, Any}(
            "optimal_config" => optimal_config,
            "performance_metrics" => performance_metrics,
            "metadata" => Dict{String, Any}(
                "generated_at" => string(now()),
                "methodology" => "PSO_plus_LBFGS_hybrid_bayesian_optimization",
                "cv_method" => "cross_regime_walk_forward_30days"
            )
        )
        
        open("optimal_config.toml", "w") do io
            TOML.print(io, output_data)
        end
        println("✅ Configuração ótima salva em: optimal_config.toml")
    end

    println("\n🎉 Walk-forward contínuo concluído!")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end