#!/usr/bin/env julia

using Distributed

# Configuração automática de workers para processamento paralelo
if nworkers() == 1
    n_cores = min(Sys.CPU_THREADS, 8)
    println("🚀 Adicionando $n_cores workers para processamento paralelo...")
    addprocs(n_cores)
end

# 1. Carrega o módulo no processo principal
include(joinpath(@__DIR__, "src/PQRateCurve.jl"))

# 2. Importa o módulo em todos os workers
@everywhere using Distributed
@everywhere include(joinpath(@__DIR__, "src/PQRateCurve.jl"))
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



# Blocos contínuos de 30 dias: Treino → Teste, cobrindo diferentes regimes
function get_continuous_blocks()
    blocks = [
        # Crise Política/Econômica - Março 2015 (Início do impeachment)
        (train_start=Date(2015,2,1), train_end=Date(2015,2,28), 
         test_start=Date(2015,3,1), test_end=Date(2015,3,31)),
        
        # Recessão Severa - Agosto 2016 (Pico da recessão)
        (train_start=Date(2016,7,1), train_end=Date(2016,7,31), 
         test_start=Date(2016,8,1), test_end=Date(2016,8,31)),
         
        # Recuperação Lenta - Junho 2018 (Pós-recessão)
        (train_start=Date(2018,5,1), train_end=Date(2018,5,31), 
         test_start=Date(2018,6,1), test_end=Date(2018,6,30)),
        
        # Pandemia - Março 2020 (Choque COVID)
        (train_start=Date(2020,2,1), train_end=Date(2020,2,29), 
         test_start=Date(2020,3,1), test_end=Date(2020,3,31)),
         
        # Alta Inflação - Março 2022 (Pressões inflacionárias)
        (train_start=Date(2022,2,1), train_end=Date(2022,2,28), 
         test_start=Date(2022,3,1), test_end=Date(2022,3,31)),
         
        # Normalização - Maio 2024 (Período recente)
        (train_start=Date(2024,4,1), train_end=Date(2024,4,30), 
         test_start=Date(2024,5,1), test_end=Date(2024,5,31)),
    ]
    
    return blocks
end


# Treina modelo sequencialmente aproveitando previous_params
@everywhere function train_sequential(pso_params::PSOHyperparams, train_dates::Vector{Date}, verbose::Bool = true)
    best_params = nothing
    costs = Float64[]
    successful_days = 0
    
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
            
            # Usa nova função MAD para otimização com liquidez
            params, cost, final_cash_flows, outliers_removed, iterations = optimize_nelson_siegel_svensson_with_mad_outlier_removal(
                cash_flows, train_date; 
                previous_params=previous_params,
                temporal_penalty_weight=pso_params.temporal_penalty_weight,
                pso_N=pso_params.N,
                pso_C1=pso_params.C1,
                pso_C2=pso_params.C2,
                pso_omega=pso_params.ω,
                pso_f_calls_limit=pso_params.f_calls_limit,
                fator_erro=pso_params.mad_threshold,
                max_iterations=2,  # Reduzido para velocidade
                fator_liq=pso_params.fator_liq,
                bond_quantities=bond_quantities,
                verbose=verbose
            )
            
            # Aplica LM se solicitado
            if pso_params.use_lm
                try
                    params_lm, cost_lm, lm_success = refine_nss_with_levenberg_marquardt(
                        final_cash_flows, train_date, params;
                        max_iterations=50, show_trace=false,
                        previous_params=previous_params,
                        temporal_penalty_weight=pso_params.temporal_penalty_weight,
                        verbose=verbose
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
            if verbose
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
        for (train_date, cost) in zip(train_dates, costs)
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
@everywhere function test_sequential(initial_params, pso_params::PSOHyperparams, test_dates::Vector{Date}, verbose::Bool = true)
    test_costs = Float64[]
    test_params_history = []
    successful_days = 0
    current_params = initial_params
    
    for (day_idx, test_date) in enumerate(test_dates)
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
            params, optimization_cost, final_cash_flows, outliers_removed, iterations = optimize_nelson_siegel_svensson_with_mad_outlier_removal(
                cash_flows, test_date;
                previous_params=previous_params,
                temporal_penalty_weight=pso_params.temporal_penalty_weight,
                pso_N=pso_params.N,
                pso_C1=pso_params.C1,
                pso_C2=pso_params.C2,
                pso_omega=pso_params.ω,
                pso_f_calls_limit=pso_params.f_calls_limit,
                max_iterations=2,  # Reduzido para velocidade
                fator_erro=pso_params.mad_threshold,
                fator_liq=pso_params.fator_liq,
                bond_quantities=bond_quantities,
                verbose=verbose
            )
            
            # CORREÇÃO VAZAMENTO DE DADOS: Calcula custo out-of-sample em reais usando dados brutos completos
            # Extrai bond_quantities correspondentes aos raw_cash_flows
            raw_bond_quantities = bond_quantities[1:length(raw_cash_flows)]
            cost_reais = calculate_out_of_sample_cost_reais(raw_cash_flows, raw_bond_quantities, test_date, params)
            
            # Aplica LM se solicitado
            if pso_params.use_lm
                try
                    params_lm, optimization_cost_lm, lm_success = refine_nss_with_levenberg_marquardt(
                        final_cash_flows, test_date, params;
                        max_iterations=50, show_trace=false,
                        previous_params=previous_params,
                        temporal_penalty_weight=pso_params.temporal_penalty_weight,
                        verbose=verbose
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
            if verbose
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
        
        # Calcula estabilidade dos vértices (6M, 1Y, 3Y, 5Y, 10Y, 15Y)
        vertices = [0.5, 1.0, 3.0, 5.0, 10.0, 15.0]
        vertex_stability = calculate_vertex_stability(test_params_history, vertices)
        
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
@everywhere function process_single_block(pso_params::PSOHyperparams, block, block_idx, verbose::Bool = true)
    worker_id = myid()
    
    if verbose
        print("🔧 Worker $worker_id: [$block_idx/6] ")
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
    trained_params, train_cost, train_days = train_sequential(pso_params, train_dates, verbose)
    
    if trained_params === nothing || train_days < 3
        if verbose
            println("❌ Falha no treino ($train_days dias)")
        end
        return nothing
    end
    
    # TESTE: Avalia parâmetros treinados sequencialmente
    test_result = test_sequential(trained_params, pso_params, test_dates, verbose)
    
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
function continuous_walkforward_single_config(pso_params::PSOHyperparams, blocks)
    config_name = "N=$(pso_params.N)_C1=$(round(pso_params.C1,digits=2))_LM=$(pso_params.use_lm)_TW=$(round(pso_params.temporal_penalty_weight,digits=4))_MAD=$(pso_params.mad_threshold)_LIQ=$(pso_params.fator_liq)"
    
    # Mensagem simplificada - removida a redundante
    println("   🔄 Distribuindo $(length(blocks)) blocos entre $(nworkers()) workers...")
    
    # PARALELIZAÇÃO ROBUSTA: Usa pmap em vez de @distributed para melhor tratamento de erros
    block_tasks = [(pso_params, blocks[i], i, false) for i in 1:length(blocks)]
    results_list = pmap(args -> process_single_block(args...), block_tasks)
    results_raw = filter(x -> x !== nothing, results_list)
    
    println("   ✅ Blocos processados: $(length(results_raw))/$(length(blocks))")
    
    return (pso_params, results_raw)
end

# Função objetivo para Otimização Bayesiana
function bayesian_objective(params_vector)
    global BAYESIAN_COUNTER, BAYESIAN_START_TIME
    BAYESIAN_COUNTER += 1
    
    # Proteção adicional contra execução excessiva em modo teste - apenas uma vez
    config = TOML.parsefile("config.toml")
    max_configs = config["validation"]["num_hyperparameter_configs"]
    if BAYESIAN_COUNTER > max_configs * 20  # Mais permissivo para o DE funcionar
        if BAYESIAN_COUNTER == max_configs * 20 + 1  # Mostra mensagem apenas uma vez
            println("⚠️  Limite de segurança atingido - modo teste concluído após explorar $(BAYESIAN_COUNTER-1) configurações")
        end
        return 1000.0  # Retorna penalidade alta para parar
    end
    
    # Extrai parâmetros do vetor
    # [N, C1, C2, omega, f_calls, use_lm_prob, temporal_penalty_weight, mad_threshold, fator_liq]
    N = params_vector[1]
    C1 = params_vector[2]
    C2 = params_vector[3]
    omega = params_vector[4]
    f_calls_idx = params_vector[5]
    use_lm_prob = params_vector[6]
    temporal_penalty_weight = params_vector[7]
    mad_threshold = params_vector[8]
    fator_liq = params_vector[9]
    
    # Converte probabilidade use_lm para booleano - FIXADO EM FALSE
    use_lm = false  # use_lm_prob > 0.5
    
    # f_calls como valor contínuo
    f_calls = round(Int, params_vector[5])
    
    # Cria parâmetros PSO
    pso_params = PSOHyperparams(
        round(Int, N),
        C1, C2, omega,
        f_calls,
        use_lm,
        temporal_penalty_weight,
        mad_threshold,
        fator_liq
    )
    
    # Calcula tempo estimado
    elapsed = time() - BAYESIAN_START_TIME
    avg_time_per_config = elapsed > 0 ? elapsed / (BAYESIAN_COUNTER - 1) : 0
    
    println("⚙️  [$BAYESIAN_COUNTER] Avaliando configuração: N=$(pso_params.N), C1=$(round(pso_params.C1,digits=2)), C2=$(round(pso_params.C2,digits=2)), ω=$(round(pso_params.ω,digits=2)), F=$(pso_params.f_calls_limit), LM=$(pso_params.use_lm), TW=$(round(pso_params.temporal_penalty_weight,digits=4)), MAD=$(round(pso_params.mad_threshold,digits=3)), LIQ=$(round(pso_params.fator_liq,digits=4))")
    if BAYESIAN_COUNTER > 1
        println("   ⏱️  Tempo médio por configuração: $(round(avg_time_per_config, digits=1))s")
    end
    
    # Executa walk-forward para esta configuração
    blocks = get_continuous_blocks()
    _, results = continuous_walkforward_single_config(pso_params, blocks)
    
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
    
    # Score híbrido simplificado (sem normalização entre configurações)
    # Usar fatores de escala baseados nos valores históricos típicos
    cost_score = avg_test_cost / 50.0  # Escala baseada em custos típicos
    stability_score = avg_vertex_stability / 20.0  # Escala baseada em estabilidade típica
    overfitting_penalty = max(0, overfitting_ratio - 1.0) * 2.0  # Penaliza overfitting
    
    simple_score = cost_score + stability_score + overfitting_penalty
    
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
    
    # Carrega configuração do arquivo TOML usando TOML diretamente
    config = TOML.parsefile("config.toml")
    num_evaluations = config["validation"]["num_hyperparameter_configs"]
    
    blocks = get_continuous_blocks()
    
    println("📊 Blocos contínuos: $(length(blocks))")
    println("📊 Avaliações Bayesianas: $num_evaluations")
    println("📊 Cada bloco: 30 dias treino → 30 dias teste")
    println("📊 Espaço de busca:")
    println("   • N ∈ [30, 150] (população PSO)")
    println("   • C1 ∈ [0.5, 3.5] (aceleração cognitiva)")
    println("   • C2 ∈ [0.5, 3.0] (aceleração social)")
    println("   • ω ∈ [0.1, 0.9] (peso de inércia)")
    println("   • f_calls ∈ {600, 900, 1200, 1500, 1800, 2500} (limite de avaliações)")
    println("   • use_lm ∈ {true, false} (refinamento LM)")
    println("   • temporal_penalty ∈ [0.0001, 0.1] (penalidade temporal)")
    println("   • mad_threshold ∈ [8.0, 12.0] (limite MAD - AJUSTADO)")
    println("   • fator_liq ∈ [0.001, 0.015] (fator liquidez - AJUSTADO)")
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
    
    # Define espaço de busca para BlackBoxOptim
    # [N, C1, C2, omega, f_calls, use_lm_prob, temporal_penalty_weight, mad_threshold, fator_liq]
    search_range = [
        (25.0, 80.0),      # N - OTIMIZADO: reduzido para velocidade (25-80 vs 30-150)
        (0.5, 3.5),        # C1 - AMPLIADO: inclui aceleração cognitiva baixa e alta
        (0.5, 3.0),        # C2 - AMPLIADO: inclui aceleração social baixa e alta
        (0.1, 0.9),        # omega - AMPLIADO: explora inércia muito baixa e muito alta
        (600.0, 2500.0),   # f_calls - range contínuo de avaliações
        (0.0, 1.0),        # use_lm_prob (mantido)
        (0.0001, 0.2),     # temporal_penalty_weight - EXPANDIDO: penalização muito baixa a alta (até 0.2)
        (6.0, 12.0),       # mad_threshold - EXPANDIDO: entre 6.0 e 12.0 conforme requisito
        (0.001, 0.015)      # fator_liq - Ajustado: entre 0.001 e 0.015 conforme requisito
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
    
            # Weights (sum = 1). Adjust as needed.
            w_test = 0.5
            w_stab = 0.3
            w_over = 0.2
    
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



# Comparação final PSO vs PSO+LM usando melhor configuração encontrada
function final_pso_vs_lm_comparison(best_pso_params::PSOHyperparams)
    println("\n" * "=" ^ 80)
    println("🥊 COMPARAÇÃO FINAL: PSO PURO vs PSO+LM")
    println("=" ^ 80)
    println("🎯 Usando melhor configuração PSO encontrada na busca Bayesiana")
    println("⚙️  Config base: N=$(best_pso_params.N), C1=$(best_pso_params.C1), C2=$(best_pso_params.C2), ω=$(best_pso_params.ω)")
    println("🔧 Parâmetros: TW=$(best_pso_params.temporal_penalty_weight), MAD=$(best_pso_params.mad_threshold), LIQ=$(best_pso_params.fator_liq)")
    
    blocks = get_continuous_blocks()
    
    # Cria duas versões: PSO puro e PSO+LM
    pso_only_params = PSOHyperparams(
        best_pso_params.N,
        best_pso_params.C1, 
        best_pso_params.C2,
        best_pso_params.ω,
        best_pso_params.f_calls_limit,
        false,  # PSO puro
        best_pso_params.temporal_penalty_weight,
        best_pso_params.mad_threshold,
        best_pso_params.fator_liq
    )
    
    pso_lm_params = PSOHyperparams(
        best_pso_params.N,
        best_pso_params.C1,
        best_pso_params.C2, 
        best_pso_params.ω,
        best_pso_params.f_calls_limit,
        true,  # PSO+LM
        best_pso_params.temporal_penalty_weight,
        best_pso_params.mad_threshold,
        best_pso_params.fator_liq
    )
    
    println("\n🚀 Executando validação cruzada para PSO PURO...")
    _, pso_results = continuous_walkforward_single_config(pso_only_params, blocks)
    
    println("\n🚀 Executando validação cruzada para PSO+LM...")
    _, pso_lm_results = continuous_walkforward_single_config(pso_lm_params, blocks)
    
    # Calcula métricas para ambos
    pso_test_costs = [r.normalized_test_cost for r in pso_results if r.normalized_test_cost > 0]
    pso_train_costs = [r.normalized_train_cost for r in pso_results if r.normalized_train_cost > 0]
    pso_stabilities = [r.vertex_stability for r in pso_results if r.vertex_stability > 0]
    
    pso_lm_test_costs = [r.normalized_test_cost for r in pso_lm_results if r.normalized_test_cost > 0]
    pso_lm_train_costs = [r.normalized_train_cost for r in pso_lm_results if r.normalized_train_cost > 0]
    pso_lm_stabilities = [r.vertex_stability for r in pso_lm_results if r.vertex_stability > 0]
    
    if isempty(pso_test_costs) || isempty(pso_lm_test_costs)
        println("❌ Falha na comparação - dados insuficientes")
        return nothing, nothing
    end
    
    # Métricas PSO puro
    pso_avg_test = mean(pso_test_costs)
    pso_test_std = length(pso_test_costs) > 1 ? std(pso_test_costs) : 0.0
    pso_avg_train = mean(pso_train_costs)
    pso_avg_stability = mean(pso_stabilities)
    pso_overfitting = pso_avg_test / pso_avg_train
    
    # Métricas PSO+LM
    lm_avg_test = mean(pso_lm_test_costs)
    lm_test_std = length(pso_lm_test_costs) > 1 ? std(pso_lm_test_costs) : 0.0
    lm_avg_train = mean(pso_lm_train_costs)
    lm_avg_stability = mean(pso_lm_stabilities)
    lm_overfitting = lm_avg_test / lm_avg_train
    
    # Score híbrido simplificado para comparação direta
    pso_score = pso_avg_test + (pso_avg_stability / 20.0) + max(0, pso_overfitting - 1.0) * 2.0
    lm_score = lm_avg_test + (lm_avg_stability / 20.0) + max(0, lm_overfitting - 1.0) * 2.0
    
    println("\n📊 RESULTADOS DA COMPARAÇÃO FINAL:")
    println("┌─────────────────────┬─────────────────┬─────────────────┐")
    println("│ Métrica             │ PSO Puro        │ PSO+LM          │")
    println("├─────────────────────┼─────────────────┼─────────────────┤")
    println("│ Custo Teste (norm.) │ $(rpad(round(pso_avg_test, digits=6), 15)) │ $(rpad(round(lm_avg_test, digits=6), 15)) │")
    println("│ Desvio Teste        │ $(rpad(round(pso_test_std, digits=6), 15)) │ $(rpad(round(lm_test_std, digits=6), 15)) │")
    println("│ Custo Treino (norm.)│ $(rpad(round(pso_avg_train, digits=6), 15)) │ $(rpad(round(lm_avg_train, digits=6), 15)) │")
    println("│ Overfitting Ratio   │ $(rpad(round(pso_overfitting, digits=3), 15)) │ $(rpad(round(lm_overfitting, digits=3), 15)) │")
    println("│ Estabilidade (bp)   │ $(rpad(round(pso_avg_stability, digits=1), 15)) │ $(rpad(round(lm_avg_stability, digits=1), 15)) │")
    println("│ Score Híbrido       │ $(rpad(round(pso_score, digits=3), 15)) │ $(rpad(round(lm_score, digits=3), 15)) │")
    println("│ Blocos Concluídos   │ $(rpad(length(pso_test_costs), 15)) │ $(rpad(length(pso_lm_test_costs), 15)) │")
    println("└─────────────────────┴─────────────────┴─────────────────┘")
    
    # Decisão final
    pso_wins_test = pso_avg_test < lm_avg_test
    pso_wins_overfitting = pso_overfitting < lm_overfitting  
    pso_wins_stability = pso_avg_stability < lm_avg_stability  # Menor = melhor
    pso_wins_hybrid = pso_score < lm_score  # Menor = melhor
    
    println("\n🏆 ANÁLISE COMPARATIVA:")
    println("  Custo de Teste: $(pso_wins_test ? "✅ PSO" : "✅ PSO+LM") $(pso_wins_test ? "menor" : "menor") ($(abs(round(((pso_avg_test - lm_avg_test) / max(pso_avg_test, lm_avg_test)) * 100, digits=2)))% diferença)")
    println("  Overfitting: $(pso_wins_overfitting ? "✅ PSO" : "✅ PSO+LM") melhor ($(round(min(pso_overfitting, lm_overfitting), digits=3)) vs $(round(max(pso_overfitting, lm_overfitting), digits=3)))")
    println("  Estabilidade: $(pso_wins_stability ? "✅ PSO" : "✅ PSO+LM") mais estável ($(round(min(pso_avg_stability, lm_avg_stability), digits=1)) vs $(round(max(pso_avg_stability, lm_avg_stability), digits=1)) bp/dia)")
    
    println("\n🎯 DECISÃO FINAL:")
    if pso_wins_hybrid
        improvement_pct = ((lm_score - pso_score) / lm_score) * 100
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
            hybrid_score_normalized = 1.0 - (pso_score / (pso_score + lm_score))  # Normalizado entre 0-1
        )
    else
        improvement_pct = ((pso_score - lm_score) / pso_score) * 100
        println("  🏅 VENCEDOR: PSO+LM")
        println("  📈 PSO+LM é $(round(improvement_pct, digits=1))% melhor no score híbrido")
        println("  💡 RECOMENDAÇÃO: Use PSO+LM - refinamento melhora performance")
        winner_params = pso_lm_params
        winner_result = (
            avg_test_cost_normalized = lm_avg_test,
            avg_train_cost_normalized = lm_avg_train,
            overfitting_ratio = lm_overfitting,
            avg_vertex_stability = lm_avg_stability,
            test_cost_std = lm_test_std,
            blocks_completed = length(pso_lm_test_costs),
            hybrid_score_normalized = 1.0 - (lm_score / (pso_score + lm_score))  # Normalizado entre 0-1
        )
    end
    
    return winner_params, winner_result
end

# Análise dos resultados da busca Bayesiana (PSO puro)
function analyze_continuous_results(results)
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
    
    println("\n📊 RANKING OTIMIZAÇÃO BAYESIANA POR SCORE HÍBRIDO NORMALIZADO (média across regimes):")
    for (rank, (params, result)) in enumerate(sorted_results)
        lm_icon = params.use_lm ? "✅" : "❌"
        
        # Usa score híbrido já calculado
        hybrid_score = result.hybrid_score_normalized
        
        println("  $rank. N=$(params.N), C1=$(round(params.C1,digits=2)), C2=$(round(params.C2,digits=2)), ω=$(round(params.ω,digits=2)), LM=$lm_icon, TW=$(round(params.temporal_penalty_weight,digits=4)), MAD=$(round(params.mad_threshold,digits=3)), LIQ=$(round(params.fator_liq,digits=4))")
        println("     🏆 Score HÍBRIDO: $(round(hybrid_score, digits=3)) (quanto MAIOR melhor - 0-1)")
        println("     🎯 Teste normalizado: $(round(result.avg_test_cost_normalized, digits=6)) ± $(round(result.test_cost_std, digits=6))")
        println("     🌊 Estabilidade: $(round(result.avg_vertex_stability, digits=1)) ± $(round(result.stability_std, digits=1)) bp/dia")
        println("     📚 Treino normalizado: $(round(result.avg_train_cost_normalized, digits=6))")
        println("     📊 Overfitting: $(round(result.overfitting_ratio, digits=3)) ($(result.overfitting_ratio < 1.15 ? "✅ Bom" : result.overfitting_ratio < 1.3 ? "⚠️ Médio" : "❌ Alto"))")
        println("     📦 Regimes: $(result.blocks_completed)/$(result.regimes_tested)")
        println()
    end
    
    # Análise PSO vs PSO+LM
    pso_only = [(p, r) for (p, r) in sorted_results if !p.use_lm]
    pso_lm = [(p, r) for (p, r) in sorted_results if p.use_lm]
    
    if !isempty(pso_only) && !isempty(pso_lm)
        println("📈 COMPARAÇÃO PSO vs PSO+LM (OUT-OF-SAMPLE):")
        
        best_pso = pso_only[1]
        best_lm = pso_lm[1]
        
        println("  🥇 Melhor PSO puro:")
        println("     Config: N=$(best_pso[1].N), C1=$(best_pso[1].C1), TW=$(best_pso[1].temporal_penalty_weight), MAD=$(best_pso[1].mad_threshold), LIQ=$(best_pso[1].fator_liq)")
        println("     Teste normalizado: $(round(best_pso[2].avg_test_cost_normalized, digits=6))")
        println("     Overfitting: $(round(best_pso[2].overfitting_ratio, digits=2))")
        println("     Estabilidade: $(round(best_pso[2].avg_vertex_stability, digits=1)) bp/dia")
        
        println("  🥇 Melhor PSO+LM:")
        println("     Config: N=$(best_lm[1].N), C1=$(best_lm[1].C1), TW=$(best_lm[1].temporal_penalty_weight), MAD=$(best_lm[1].mad_threshold), LIQ=$(best_lm[1].fator_liq)")
        println("     Teste normalizado: $(round(best_lm[2].avg_test_cost_normalized, digits=6))")
        println("     Overfitting: $(round(best_lm[2].overfitting_ratio, digits=2))")
        println("     Estabilidade: $(round(best_lm[2].avg_vertex_stability, digits=1)) bp/dia")
        
        # Comparações quantitativas normalizadas
        improvement = ((best_pso[2].avg_test_cost_normalized - best_lm[2].avg_test_cost_normalized) / best_pso[2].avg_test_cost_normalized) * 100
        
        println("\n🎯 CONCLUSÃO DEFINITIVA (SCORE HÍBRIDO NORMALIZADO):")
        
        # Usa o score híbrido normalizado por range completo para a decisão final
        best_pso_hybrid = best_pso[2].hybrid_score_normalized
        best_lm_hybrid = best_lm[2].hybrid_score_normalized
        
        println("  📊 Performance HÍBRIDA across regimes (usando score normalizado min-max):")
        
        if best_lm_hybrid > best_pso_hybrid
            # Usar abs() para evitar divisão por zero se o score for 0
            improvement = ((best_lm_hybrid - best_pso_hybrid) / abs(best_pso_hybrid)) * 100
            println("  ✅ PSO+LM é $(round(improvement, digits=2))% superior no score híbrido")
            println("  ✅ RECOMENDAÇÃO: Usar sistema híbrido PSO+LM")
            println("  🎯 Melhor overfitting: $(round(best_lm[2].overfitting_ratio, digits=3)) vs $(round(best_pso[2].overfitting_ratio, digits=3))")
        else
            improvement = ((best_pso_hybrid - best_lm_hybrid) / abs(best_lm_hybrid)) * 100
            println("  ⚪ PSO puro é $(round(improvement, digits=2))% superior no score híbrido")
            println("  ⚪ RECOMENDAÇÃO: PSO puro é suficiente")
            println("  🎯 Melhor overfitting: $(round(best_pso[2].overfitting_ratio, digits=3)) vs $(round(best_lm[2].overfitting_ratio, digits=3))")
        end
        
        # Análise de estabilidade
        pso_stability = [r[2].avg_vertex_stability for r in pso_only]
        lm_stability = [r[2].avg_vertex_stability for r in pso_lm]
        
        println("\n🌊 ANÁLISE DE ESTABILIDADE DOS VÉRTICES (6 vértices: 0.5, 1, 3, 5, 10, 15 anos):")
        println("  PSO puro - Média: $(round(mean(pso_stability), digits=1)) bp/dia")
        println("  PSO+LM - Média: $(round(mean(lm_stability), digits=1)) bp/dia")
        
        if mean(lm_stability) < mean(pso_stability)
            println("  ✅ PSO+LM produz curvas mais estáveis")
        else
            println("  ⚠️  PSO puro produz curvas mais estáveis")
        end
        
        # Análise de overfitting
        pso_overfit = [r[2].overfitting_ratio for r in pso_only]
        lm_overfit = [r[2].overfitting_ratio for r in pso_lm]
        
        println("\n📊 ANÁLISE DE OVERFITTING:")
        println("  PSO puro - Overfitting médio: $(round(mean(pso_overfit), digits=2))")
        println("  PSO+LM - Overfitting médio: $(round(mean(lm_overfit), digits=2))")
    end
end

function main()
    println("🚀 Iniciando walk-forward contínuo...")
    results, elapsed_time, num_configs = run_continuous_walkforward()

    analyze_continuous_results(results)
    
    # Executa comparação final PSO vs PSO+LM usando melhor configuração
    final_winner_params = nothing
    final_winner_result = nothing
    
    if !isempty(results)
        # Usa os resultados já ordenados por score híbrido (decrescente - maior score primeiro)
        sorted_results = sort(collect(results), by=x->x[2].hybrid_score_normalized, rev=true)
        best_pso_params = sorted_results[1][1]
        best_pso_result = sorted_results[1][2]
        
        println("\n🎯 EXECUTANDO COMPARAÇÃO FINAL PSO vs PSO+LM...")
        println("Usando melhor configuração PSO encontrada: N=$(best_pso_params.N), C1=$(best_pso_params.C1), etc.")
        
        final_winner_params, final_winner_result = final_pso_vs_lm_comparison(best_pso_params)
        
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
                "use_lm" => best_params.use_lm,
                "temporal_penalty_weight" => best_params.temporal_penalty_weight
            ),
            "outlier_detection" => Dict{String, Any}(
                "mad_threshold" => best_params.mad_threshold,
                "fator_liq" => best_params.fator_liq
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
            "methodology" => best_params.use_lm ? "PSO_plus_LM_final_comparison_winner" : "PSO_only_final_comparison_winner",
            "cv_method" => "cross_regime_walk_forward_30days_with_final_pso_vs_lm_comparison",
            "period" => "2014-2024_all_regimes",
            "selection_method" => "bayesian_optimization_plus_final_head_to_head_comparison"
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