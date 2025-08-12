#!/usr/bin/env julia

using Distributed, TOML, Logging

# Configuração de workers para processamento paralelo a partir do config.toml
function setup_workers()
    config = TOML.parsefile("config.toml")
    cv_config = get(config, "cross_validation", Dict())
    max_cores = get(cv_config, "max_cores", Sys.CPU_THREADS)

    if nworkers() == 1
        n_cores = min(Sys.CPU_THREADS, max_cores)
        println("🚀 Adicionando $n_cores workers para processamento paralelo...")
        addprocs(n_cores)
    end
end

setup_workers()

# 1. Carrega o módulo no processo principal (apenas uma vez, suprimindo warnings de docs)
if !isdefined(Main, :PQRateCurve)
    with_logger(NullLogger()) do
        include(joinpath(@__DIR__, "src/PQRateCurve.jl"))
    end
end

# 2. Importa o módulo em todos os workers (suprimindo warnings de docs)
@everywhere using Distributed, Logging
@everywhere if !isdefined(Main, :PQRateCurve)
    with_logger(NullLogger()) do
        include(joinpath(@__DIR__, "src/PQRateCurve.jl"))
    end
end
@everywhere using Main.PQRateCurve

# 3. Carrega dependências básicas em todos os processos
using Dates, Statistics, DataFrames, TOML
@everywhere using Dates, Statistics, DataFrames, TOML, CSV, HTTP, ZipFile, LinearAlgebra, Random, Optim, JSON

# 4. Adiciona Metaheuristics para Otimização
using Metaheuristics
@everywhere using Metaheuristics


println("🎯 WALK-FORWARD CONTÍNUO PSO+LM - Blocos de 30 dias")
println("=" ^ 60)
println("🔄 Modo: PARALELO ($(nworkers()) workers)")

# Gera todas as datas úteis em um período
@everywhere function get_business_dates(start_date::Date, end_date::Date)
    dates = Date[]
    current = start_date
    
    while current <= end_date
        weekday = Dates.dayofweek(current)
        if 1 <= weekday <= 5  # Segunda a sexta
            push!(dates, current)
        end
        current += Day(1)
    end
    
    return dates
end

@everywhere struct PSOHyperparams
    N::Int
    C1::Float64
    C2::Float64
    ω::Float64
    f_calls_limit::Int
    use_lm::Bool
    temporal_penalty_weight::Float64
    mad_threshold::Float64
    fator_liq::Float64
end

# Cache global para armazenar resultados de todas as avaliações da Otimização Bayesiana
global BAYESIAN_RESULTS = Vector{Any}()
global BAYESIAN_COUNTER = 0
global BAYESIAN_START_TIME = 0.0



# Carrega blocos de treino/teste a partir do config.toml
function get_continuous_blocks_from_config()
    config = TOML.parsefile("config.toml")
    cv_config = get(config, "cross_validation", Dict())
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
@everywhere function train_sequential(pso_params::PSOHyperparams, train_dates::Vector{Date}, train_max_iterations::Int, lm_max_iterations_cv::Int, log_level::Int = 1)
    best_params = nothing
    costs = Float64[]
    successful_days = 0
    
    # Load bounds from config file inside the worker
    config = TOML.parsefile("config.toml")
    pso_config = get(config, "pso", Dict())
    lower_bounds = get(pso_config, "lower_bounds", [0.01, -0.25, -0.30, -0.20, 0.5, 2.0])
    upper_bounds = get(pso_config, "upper_bounds", [0.30, 0.25, 0.30, 0.20, 20.0, 50.0])

    for (day_idx, train_date) in enumerate(train_dates)
        try
            df = load_bacen_data(train_date, train_date)
            
            if nrow(df) < 3
                if log_level >= 2
                    println("❌ Dados insuficientes para $train_date: $(nrow(df)) linhas")
                end
                continue
            end
            
            # Aproveita parâmetros do dia anterior
            previous_params = (day_idx > 1 && best_params !== nothing) ? best_params : nothing
            
            # Gera cash flows com informação de quantidade para MAD híbrido
            cash_flows, bond_quantities, _ = generate_cash_flows_with_quantity(df, train_date)
            
            # Usa nova função MAD para otimização com liquidez
            params, cost, final_cash_flows, _, _ = optimize_nelson_siegel_svensson_with_mad_outlier_removal(
                cash_flows, train_date, lower_bounds, upper_bounds;
                previous_params=previous_params,
                temporal_penalty_weight=pso_params.temporal_penalty_weight,
                pso_N=pso_params.N,
                pso_C1=pso_params.C1,
                pso_C2=pso_params.C2,
                pso_omega=pso_params.ω,
                pso_f_calls_limit=pso_params.f_calls_limit,
                fator_erro=pso_params.mad_threshold,
                max_iterations=train_max_iterations,
                fator_liq=pso_params.fator_liq,
                bond_quantities=bond_quantities,
                log_level=log_level
            )
            
            # Aplica LM se solicitado
            if pso_params.use_lm
                try
                    params_lm, cost_lm, lm_success = refine_nss_with_lbfgs(
                        final_cash_flows, train_date, params, lower_bounds, upper_bounds;
                        max_iterations=lm_max_iterations_cv, show_trace=false,
                        previous_params=previous_params,
                        temporal_penalty_weight=pso_params.temporal_penalty_weight,
                        log_level=log_level
                    )
                    
                    if lm_success && cost_lm < cost
                        params = params_lm
                        cost = cost_lm
                    end
                catch
                    # Mantém PSO se LM falhar
                end
            end
            
            best_params = params
            
            # Calcula custo out-of-sample em reais para o dia de treino
            cost_reais = calculate_out_of_sample_cost_reais(final_cash_flows, bond_quantities[1:length(final_cash_flows)], train_date, params)
            push!(costs, cost_reais)
            successful_days += 1
            
        catch e
            if log_level >= 1
                println("❌ Erro em $train_date: $e")
            end
            continue
        end
    end
    
    if successful_days > 0 && best_params !== nothing
        # Soma os custos diários em MÓDULO (valor absoluto)
        total_train_cost_reais = sum(abs.(costs))
        
        # Calcula volume total negociado no período de treino para normalização
        total_train_volume = 0.0
        for (train_date, _) in zip(train_dates, costs)
            try
                df = load_bacen_data(train_date, train_date)
                if nrow(df) >= 3
                    _, bond_quantities, _ = generate_cash_flows_with_quantity(df, train_date)
                    total_train_volume += sum(bond_quantities)
                end
            catch
                continue
            end
        end
        
        # Normaliza pelo volume total (custo por unidade de volume)
        normalized_train_cost = total_train_volume > 0 ? total_train_cost_reais / total_train_volume : total_train_cost_reais
        
        return best_params, normalized_train_cost, successful_days
    else
        return nothing, Inf, 0
    end
end

# Testa modelo treinado sequencialmente no período de teste
@everywhere function test_sequential(initial_params, pso_params::PSOHyperparams, test_dates::Vector{Date}, train_max_iterations::Int, lm_max_iterations_cv::Int, stability_vertices::Vector{Float64}, log_level::Int = 1)
    test_costs = Float64[]
    test_params_history = []
    successful_days = 0
    current_params = initial_params

    # Load bounds from config file inside the worker
    config = TOML.parsefile("config.toml")
    pso_config = get(config, "pso", Dict())
    lower_bounds = get(pso_config, "lower_bounds", [0.01, -0.25, -0.30, -0.20, 0.5, 2.0])
    upper_bounds = get(pso_config, "upper_bounds", [0.30, 0.25, 0.30, 0.20, 20.0, 50.0])
    
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
            
            # Usa nova função MAD para otimização com liquidez (obtém parâmetros)
            params, optimization_cost, final_cash_flows, _, _ = optimize_nelson_siegel_svensson_with_mad_outlier_removal(
                cash_flows, test_date, lower_bounds, upper_bounds;
                previous_params=previous_params,
                temporal_penalty_weight=pso_params.temporal_penalty_weight,
                pso_N=pso_params.N,
                pso_C1=pso_params.C1,
                pso_C2=pso_params.C2,
                pso_omega=pso_params.ω,
                pso_f_calls_limit=pso_params.f_calls_limit,
                max_iterations=train_max_iterations,
                fator_erro=pso_params.mad_threshold,
                fator_liq=pso_params.fator_liq,
                bond_quantities=bond_quantities,
                log_level=log_level
            )
            
            # CORREÇÃO VAZAMENTO DE DADOS: Calcula custo out-of-sample em reais usando dados brutos completos
            # Extrai bond_quantities correspondentes aos raw_cash_flows
            raw_bond_quantities = bond_quantities[1:length(raw_cash_flows)]
            cost_reais = calculate_out_of_sample_cost_reais(raw_cash_flows, raw_bond_quantities, test_date, params)
            
            # Aplica LM se solicitado
            if pso_params.use_lm
                try
                    params_lm, _, lm_success = refine_nss_with_lbfgs(
                        final_cash_flows, test_date, params, lower_bounds, upper_bounds;
                        max_iterations=lm_max_iterations_cv, show_trace=false,
                        previous_params=previous_params,
                        temporal_penalty_weight=pso_params.temporal_penalty_weight,
                        log_level=log_level
                    )
                    
                    # CORREÇÃO VAZAMENTO DE DADOS: Avalia LM usando dados brutos completos em reais
                    cost_lm_reais = calculate_out_of_sample_cost_reais(raw_cash_flows, raw_bond_quantities, test_date, params_lm)
                    
                    # Compara custos absolutos em reais 
                    if lm_success && abs(cost_lm_reais) < abs(cost_reais)
                        params = params_lm
                        cost_reais = cost_lm_reais
                    end
                catch
                    # Mantém PSO se LM falhar
                end
            end
            
            current_params = params
            push!(test_costs, cost_reais)
            push!(test_params_history, current_params)
            successful_days += 1
            
        catch e
            if log_level >= 1
                println("❌ Erro no teste sequencial em $test_date: $e. Interrompendo bloco.")
            end
            break
        end
    end
    
    if successful_days > 0
        # Soma os custos diários em MÓDULO (valor absoluto) - total do bloco de 30 dias
        total_test_cost_reais = sum(abs.(test_costs))
        
        # Calcula volume total negociado no período de teste para normalização
        total_test_volume = 0.0
        for (test_date, cost) in zip(test_dates, test_costs)
            try
                df = load_bacen_data(test_date, test_date)
                if nrow(df) >= 3
                    _, bond_quantities, _ = generate_cash_flows_with_quantity(df, test_date)
                    total_test_volume += sum(bond_quantities)
                end
            catch
                continue
            end
        end
        
        # Normaliza pelo volume total (custo por unidade de volume)
        normalized_test_cost = total_test_volume > 0 ? total_test_cost_reais / total_test_volume : total_test_cost_reais
        
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
@everywhere function process_single_block(pso_params::PSOHyperparams, block, block_idx, cv_config::Dict, log_level::Int = 1)
    worker_id = myid()
    
    train_max_iterations = get(cv_config, "train_max_iterations", 2)
    lm_max_iterations_cv = get(cv_config, "lm_max_iterations_cv", 50)
    stability_vertices = get(cv_config, "stability_vertices", [0.5, 1.0, 3.0, 5.0, 10.0, 15.0])

    if log_level >= 1
        print("🔧 Worker $worker_id: [$block_idx/$(length(cv_config["blocks"]))] ")
        print("Treino: $(block.train_start) a $(block.train_end), ")
        print("Teste: $(block.test_start) a $(block.test_end) ")
    end
    
    # Gera datas úteis
    train_dates = get_business_dates(block.train_start, block.train_end)
    test_dates = get_business_dates(block.test_start, block.test_end)
    
    if isempty(train_dates) || isempty(test_dates)
        if log_level >= 1
            println("❌ Sem datas úteis")
        end
        return nothing
    end
    
    # TREINO: Sequencial com previous_params
    trained_params, train_cost, train_days = train_sequential(pso_params, train_dates, train_max_iterations, lm_max_iterations_cv, log_level)
    
    if trained_params === nothing || train_days < 3
        if log_level >= 1
            println("❌ Falha no treino ($train_days dias)")
        end
        return nothing
    end
    
    # TESTE: Avalia parâmetros treinados sequencialmente
    test_result = test_sequential(trained_params, pso_params, test_dates, train_max_iterations, lm_max_iterations_cv, stability_vertices, log_level)
    
    if test_result === nothing
        if log_level >= 1
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
    
    if log_level >= 1
        println("✅ Treino: $(round(train_cost, digits=6)) norm ($(train_days)d), Teste: $(round(test_result.normalized_test_cost, digits=6)) norm ($(test_result.successful_days)d), Estab: $(round(test_result.vertex_stability, digits=1))bp")
    end
    
    return result
end

# Walk-forward para uma configuração PSO - VERSÃO PARALELA
function continuous_walkforward_single_config(pso_params::PSOHyperparams, blocks, cv_config::Dict, log_level::Int)
    config_name = "N=$(pso_params.N)_C1=$(round(pso_params.C1,digits=2))_LM=$(pso_params.use_lm)_TW=$(round(pso_params.temporal_penalty_weight,digits=4))_MAD=$(pso_params.mad_threshold)_LIQ=$(pso_params.fator_liq)"
    
    if log_level >= 1
        println("   🔄 Distribuindo $(length(blocks)) blocos entre $(nworkers()) workers...")
    end
    
    # PARALELIZAÇÃO ROBUSTA: Usa pmap em vez de @distributed para melhor tratamento de erros
    block_tasks = [(pso_params, blocks[i], i, cv_config, log_level) for i in 1:length(blocks)]
    results_list = pmap(args -> process_single_block(args...), block_tasks)
    results_raw = filter(x -> x !== nothing, results_list)
    
    println("   ✅ Blocos processados: $(length(results_raw))/$(length(blocks))")
    
    return (pso_params, results_raw)
end

# Helper function to calculate the simplified hybrid score for Bayesian optimization
function _calculate_bayesian_score(percentile_test_cost, avg_vertex_stability, overfitting_ratio, config)
    scaling_config = get(get(config, "cross_validation", Dict()), "bayesian_objective_scaling", Dict())
    cost_scale = get(scaling_config, "cost_score", 50.0)
    stability_scale = get(scaling_config, "stability_score", 20.0)
    overfitting_penalty_mult = get(scaling_config, "overfitting_penalty", 2.0)

    cost_score = percentile_test_cost / cost_scale
    stability_score = avg_vertex_stability / stability_scale
    overfitting_penalty = max(0, overfitting_ratio - 1.0) * overfitting_penalty_mult

    return cost_score + stability_score + overfitting_penalty
end

# Execução principal com Otimização Bayesiana
function run_continuous_walkforward()
    # --- CONFIGURAÇÃO DE LOG ---
    # Nível 1: Progresso (Padrão)
    # Nível 2: Informativo
    # Nível 3: Debug Completo
    log_level = 1
    # -------------------------

    # Função objetivo para Otimização Bayesiana (definida aqui para capturar log_level)
    function bayesian_objective(params_vector)
        global BAYESIAN_COUNTER, BAYESIAN_START_TIME
        BAYESIAN_COUNTER += 1

        config = TOML.parsefile("config.toml")
        max_configs = config["validation"]["num_hyperparameter_configs"]
        if BAYESIAN_COUNTER > max_configs * 20
            if BAYESIAN_COUNTER == max_configs * 20 + 1
                println("⚠️  Limite de segurança atingido - modo teste concluído após explorar $(BAYESIAN_COUNTER-1) configurações")
            end
            return 1000.0
        end

        N = params_vector[1]
        C1 = params_vector[2]
        C2 = params_vector[3]
        omega = params_vector[4]
        f_calls = round(Int, params_vector[5])
        use_lm_val = params_vector[6]
        temporal_penalty_weight = params_vector[7]
        mad_threshold = params_vector[8]
        fator_liq = params_vector[9]

        use_lm = use_lm_val > 0.5

        pso_params = PSOHyperparams(round(Int, N), C1, C2, omega, f_calls, use_lm, temporal_penalty_weight, mad_threshold, fator_liq)

        elapsed = time() - BAYESIAN_START_TIME
        avg_time_per_config = elapsed > 0 ? elapsed / (BAYESIAN_COUNTER - 1) : 0

        if log_level >= 1
            println("⚙️  [$BAYESIAN_COUNTER] Avaliando configuração: N=$(pso_params.N), C1=$(round(pso_params.C1,digits=2)), C2=$(round(pso_params.C2,digits=2)), ω=$(round(pso_params.ω,digits=2)), F=$(pso_params.f_calls_limit), LM=$(pso_params.use_lm), TW=$(round(pso_params.temporal_penalty_weight,digits=4)), MAD=$(round(pso_params.mad_threshold,digits=3)), LIQ=$(round(pso_params.fator_liq,digits=4))")
            if BAYESIAN_COUNTER > 1
                println("   ⏱️  Tempo médio por configuração: $(round(avg_time_per_config, digits=1))s")
            end
        end

        cv_config = get(config, "cross_validation", Dict())
        blocks = get_continuous_blocks_from_config()
        _, results = continuous_walkforward_single_config(pso_params, blocks, cv_config, log_level)

        test_costs = [r.normalized_test_cost for r in results if r.normalized_test_cost > 0]
        train_costs = [r.normalized_train_cost for r in results if r.normalized_train_cost > 0]
        vertex_stabilities = [r.vertex_stability for r in results if r.vertex_stability > 0]

        if isempty(test_costs) || isempty(train_costs) || isempty(vertex_stabilities)
            if log_level >= 1; println("❌ Configuração falhou - retornando penalidade"); end
            return 1000.0
        end

        test_cost_perc_val = get(cv_config, "test_cost_percentile", 95.0)
        percentile_test_cost = percentile(test_costs, test_cost_perc_val)

        avg_train_cost = mean(train_costs)
        avg_vertex_stability = mean(vertex_stabilities)
        stability_std = length(vertex_stabilities) > 1 ? std(vertex_stabilities) : 0.0

        overfitting_ratio = percentile_test_cost / avg_train_cost

        simple_score = _calculate_bayesian_score(percentile_test_cost, avg_vertex_stability, overfitting_ratio, config)

        result_data = Dict(
            "pso_params" => pso_params,
            "percentile_test_cost" => percentile_test_cost,
            "avg_train_cost" => avg_train_cost,
            "avg_vertex_stability" => avg_vertex_stability,
            "stability_std" => stability_std,
            "overfitting_ratio" => overfitting_ratio,
            "blocks_completed" => length(test_costs),
            "simple_score" => simple_score,
            "detailed_results" => results
        )

        push!(BAYESIAN_RESULTS, result_data)

        all_scores = [r["simple_score"] for r in BAYESIAN_RESULTS]
        best_score_so_far = minimum(all_scores)

        config_time = time() - BAYESIAN_START_TIME - (BAYESIAN_COUNTER - 1) * avg_time_per_config
        if log_level >= 1
            println("✅ [$BAYESIAN_COUNTER] Concluída em $(round(config_time, digits=1))s | Score Atual: $(round(simple_score, digits=3)) | Melhor Score: $(round(best_score_so_far, digits=3)) | Teste (p$(test_cost_perc_val)): $(round(percentile_test_cost, digits=3))")
        end

        return simple_score
    end

    println("⚙️  Configurando walk-forward contínuo com OTIMIZAÇÃO BAYESIANA...")
    
    # Carrega configuração do arquivo TOML
    config = TOML.parsefile("config.toml")
    validation_config = get(config, "validation", Dict())
    cv_config = get(config, "cross_validation", Dict())
    num_evaluations = get(validation_config, "num_hyperparameter_configs", 20)
    
    blocks = get_continuous_blocks_from_config()
    
    println("📊 Blocos contínuos: $(length(blocks))")
    println("📊 Avaliações Bayesianas: $num_evaluations")
    println("📊 Cada bloco: 30 dias treino → 30 dias teste (conforme config.toml)")
    # Load hyperparameter ranges for display
    config = TOML.parsefile("config.toml")
    hyperparams_config = get(config, "hyperparameter_search", Dict())
    
    println("📊 Espaço de busca (config.toml):")
    println("   • N ∈ [$(get(hyperparams_config, "N_min", 25)), $(get(hyperparams_config, "N_max", 80))] (população PSO)")
    println("   • C1 ∈ [$(get(hyperparams_config, "C1_min", 0.5)), $(get(hyperparams_config, "C1_max", 3.5))] (aceleração cognitiva)")
    println("   • C2 ∈ [$(get(hyperparams_config, "C2_min", 0.5)), $(get(hyperparams_config, "C2_max", 3.0))] (aceleração social)")
    println("   • ω ∈ [$(get(hyperparams_config, "omega_min", 0.1)), $(get(hyperparams_config, "omega_max", 0.9))] (peso de inércia)")
    println("   • f_calls ∈ [$(get(hyperparams_config, "f_calls_min", 600)), $(get(hyperparams_config, "f_calls_max", 2500))] (limite de avaliações)")
    println("   • use_lm ∈ [0.0, 1.0] (true if > 0.5)")
    println("   • temporal_penalty ∈ [$(get(hyperparams_config, "temporal_penalty_min", 0.0001)), $(get(hyperparams_config, "temporal_penalty_max", 0.2))] (penalidade temporal)")
    println("   • mad_threshold ∈ [$(get(hyperparams_config, "mad_threshold_min", 6.0)), $(get(hyperparams_config, "mad_threshold_max", 12.0))] (limite MAD)")
    println("   • fator_liq ∈ [$(get(hyperparams_config, "fator_liq_min", 0.001)), $(get(hyperparams_config, "fator_liq_max", 0.015))] (fator liquidez)")
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
    
    # Load hyperparameter search ranges from config.toml
    config = TOML.parsefile("config.toml")
    hyperparams_config = get(config, "hyperparameter_search", Dict())
    
    # Define search space with values from config or sensible defaults
    search_range = [
        (get(hyperparams_config, "N_min", 25.0), get(hyperparams_config, "N_max", 80.0)),
        (get(hyperparams_config, "C1_min", 0.5), get(hyperparams_config, "C1_max", 3.5)),
        (get(hyperparams_config, "C2_min", 0.5), get(hyperparams_config, "C2_max", 3.0)),
        (get(hyperparams_config, "omega_min", 0.1), get(hyperparams_config, "omega_max", 0.9)),
        (get(hyperparams_config, "f_calls_min", 600.0), get(hyperparams_config, "f_calls_max", 2500.0)),
        (0.0, 1.0), # use_lm is now a continuous value between 0 and 1
        (get(hyperparams_config, "temporal_penalty_min", 0.0001), get(hyperparams_config, "temporal_penalty_max", 0.2)),
        (get(hyperparams_config, "mad_threshold_min", 6.0), get(hyperparams_config, "mad_threshold_max", 12.0)),
        (get(hyperparams_config, "fator_liq_min", 0.001), get(hyperparams_config, "fator_liq_max", 0.015))
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
    all_test_costs = [r["percentile_test_cost"] for r in BAYESIAN_RESULTS if r["blocks_completed"] >= length(blocks)/2]
    all_train_costs = [r["avg_train_cost"] for r in BAYESIAN_RESULTS if r["blocks_completed"] >= length(blocks)/2]
    all_stabilities = [r["avg_vertex_stability"] for r in BAYESIAN_RESULTS if r["blocks_completed"] >= length(blocks)/2]
    
    if isempty(all_test_costs)
        println("❌ Nenhuma configuração teve sucesso suficiente")
        return Dict(), elapsed_time, num_evaluations
    end
    
    # Compute full ranges for inverse-percentile ranking
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
            # ---- Weighted geometric-mean of inverse-percentile ranks ----
            # Inverse-percentile rank (higher = better, so 1.0 - normalized_value)
            inv_test = 1.0 - (result["percentile_test_cost"] - test_min) / max(test_max - test_min, eps())
            inv_stab = 1.0 - (result["avg_vertex_stability"] - stab_min) / max(stab_max - stab_min, eps())
            inv_over = 1.0 - (result["overfitting_ratio"] - over_min) / max(over_max - over_min, eps())
    
            # Ensure positivity for log-space computation
            inv_test = max(inv_test, eps())
            inv_stab = max(inv_stab, eps())
            inv_over = max(inv_over, eps())
    
            # Weights from config
            weights_config = get(cv_config, "hybrid_score_weights", Dict("test" => 0.6, "stability" => 0.3, "overfitting" => 0.1))
            w_test = get(weights_config, "test", 0.6)
            w_stab = get(weights_config, "stability", 0.3)
            w_over = get(weights_config, "overfitting", 0.1)
    
            # Weighted geometric mean via log-space (avoids underflow)
            hybrid_score = exp(w_test * log(inv_test) + w_stab * log(inv_stab) + w_over * log(inv_over))
            
            final_results[result["pso_params"]] = (
                percentile_test_cost = result["percentile_test_cost"],
                avg_train_cost_normalized = result["avg_train_cost"],
                overfitting_ratio = result["overfitting_ratio"],
                avg_vertex_stability = result["avg_vertex_stability"],
                stability_std = result["stability_std"],
                blocks_completed = result["blocks_completed"],
                regimes_tested = length(blocks),
                hybrid_score_normalized = hybrid_score
            )
        end
    end
    
    return final_results, elapsed_time, num_evaluations
end



function _print_bayesian_ranking_table(sorted_results)
    config = TOML.parsefile("config.toml")
    cv_config = get(config, "cross_validation", Dict())
    test_cost_perc_val = get(cv_config, "test_cost_percentile", 95.0)

    println("\n📊 RANKING OTIMIZAÇÃO BAYESIANA POR SCORE HÍBRIDO NORMALIZADO (média across regimes):")
    for (rank, (params, result)) in enumerate(sorted_results)
        lm_icon = params.use_lm ? "✅" : "❌"
        hybrid_score = result.hybrid_score_normalized

        println("  $rank. N=$(params.N), C1=$(round(params.C1,digits=2)), C2=$(round(params.C2,digits=2)), ω=$(round(params.ω,digits=2)), LM=$lm_icon, TW=$(round(params.temporal_penalty_weight,digits=4)), MAD=$(round(params.mad_threshold,digits=3)), LIQ=$(round(params.fator_liq,digits=4))")
        println("     🏆 Score HÍBRIDO: $(round(hybrid_score, digits=3)) (quanto MAIOR melhor - 0-1)")
        println("     🎯 Teste (p$(test_cost_perc_val)): $(round(result.percentile_test_cost, digits=6))")
        println("     🌊 Estabilidade: $(round(result.avg_vertex_stability, digits=1)) ± $(round(result.stability_std, digits=1)) bp/dia")
        println("     📚 Treino normalizado: $(round(result.avg_train_cost_normalized, digits=6))")
        println("     📊 Overfitting: $(round(result.overfitting_ratio, digits=3)) ($(result.overfitting_ratio < 1.15 ? "✅ Bom" : result.overfitting_ratio < 1.3 ? "⚠️ Médio" : "❌ Alto"))")
        println("     📦 Regimes: $(result.blocks_completed)/$(result.regimes_tested)")
        println()
    end
end

# Análise dos resultados da busca Bayesiana
function analyze_continuous_results(results, _)
    if isempty(results)
        println("❌ Nenhum resultado para analisar")
        return nothing
    end
    
    println("\n" * "=" ^ 80)
    println("🏆 WALK-FORWARD CROSS-REGIME - RESULTADOS OTIMIZAÇÃO BAYESIANA")
    println("=" ^ 80)
    
    # Score híbrido: quanto MAIOR melhor (0-1, onde 1 = melhor possível)
    # Ordena por score híbrido (decrescente - maior score primeiro)
    sorted_results = sort(collect(results), by=x->x[2].hybrid_score_normalized, rev=true)
    
    _print_bayesian_ranking_table(sorted_results)
    
    # Retorna o melhor resultado para ser salvo
    return sorted_results[1]
end

function main()
    println("🚀 Iniciando walk-forward contínuo...")
    config = TOML.parsefile("config.toml")
    cv_config = get(config, "cross_validation", Dict())
    results, elapsed_time, num_configs = run_continuous_walkforward()

    best_overall_result = analyze_continuous_results(results, cv_config)
    
    # Salva resultados do modelo definitivo
    if best_overall_result !== nothing
        println("\n💾 Salvando configuração DEFINITIVA do modelo vencedor...")
        
        best_params = best_overall_result[1]
        best_result_metrics = best_overall_result[2]
        best_hybrid_score = best_result_metrics.hybrid_score_normalized
        
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
                "use_lm" => best_params.use_lm,
                "temporal_penalty_weight" => best_params.temporal_penalty_weight
            ),
            "outlier_detection" => Dict{String, Any}(
                "mad_threshold" => best_params.mad_threshold,
                "fator_liq" => best_params.fator_liq
            )
        )
        
        cv_config = get(config, "cross_validation", Dict())
        test_cost_perc_val = get(cv_config, "test_cost_percentile", 95.0)

        performance_metrics = Dict{String, Any}(
            "hybrid_score_normalized" => round(best_hybrid_score, digits=4),
            "percentile_test_cost_normalized" => round(best_result_metrics.percentile_test_cost, digits=8),
            "test_cost_percentile_used" => test_cost_perc_val,
            "avg_train_cost_normalized" => round(best_result_metrics.avg_train_cost_normalized, digits=8),
            "overfitting_ratio" => round(best_result_metrics.overfitting_ratio, digits=3),
            "vertex_stability_bp_per_day" => round(best_result_metrics.avg_vertex_stability, digits=2),
            "blocks_completed" => best_result_metrics.blocks_completed,
            "regimes_tested" => 6,  # Total regimes tested
            "execution_time_minutes" => round(elapsed_time/60, digits=1),
            "methodology" => best_params.use_lm ? "PSO_plus_LBFGS" : "PSO_only",
            "cv_method" => "cross_regime_walk_forward_30days",
            "period" => "2014-2024_all_regimes",
            "selection_method" => "bayesian_optimization_with_hybrid_score"
        )
        
        # Chama a função usando TOML diretamente para salvar
        output_data = Dict{String, Any}(
            "optimal_config" => optimal_config,
            "performance_metrics" => performance_metrics,
            "metadata" => Dict{String, Any}(
                "generated_at" => string(now()),
                "methodology" => "PSO_plus_LM_hybrid_bayesian_optimization",
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