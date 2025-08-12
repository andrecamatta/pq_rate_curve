#!/usr/bin/env julia --project=.

# Script unificado para fit de curvas NSS
# Uso: julia fit_curvas.jl [--start YYYY-MM-DD] [--end YYYY-MM-DD] [opções]
# Ou: ./fit_curvas.jl [--start YYYY-MM-DD] [--end YYYY-MM-DD] [opções]

include("src/PQRateCurve.jl")
using .PQRateCurve
using Dates, Statistics, CSV, DataFrames, TOML

# Estrutura para resultado de um dia
struct DayResult
    date::Date
    success::Bool
    params::Union{Vector{Float64}, Nothing}
    cost::Union{Float64, Nothing}
    n_bonds::Int
    outliers_removed::Int
    error_message::Union{String, Nothing}
    used_previous_params::Bool
    reoptimized::Bool  # Flag para indicar se foi re-otimizado por custo alto
end

# Configuração simples sem argumentos
function get_default_args()
    return Dict(
        "start" => "2024-01-01",
        "end" => "2024-12-31",
        "output" => "curvas_nss",
        "continuity" => true,
        "verbose" => true,
        "dry-run" => false
    )
end

# Lê configuração ótima (atualizada para usar TOML)
function read_optimal_config()
    # Primeiro tenta carregar optimal_config.toml (novo formato)
    if isfile("optimal_config.toml")
        config = TOML.parsefile("optimal_config.toml")
        return normalize_config_format(config)
    # Senão, usa o config.toml padrão
    elseif isfile("config.toml")
        config = TOML.parsefile("config.toml")
        return normalize_config_format(config)
    else
        error("Nenhum arquivo de configuração encontrado! (optimal_config.toml ou config.toml)")
    end
end

# Normaliza formato da configuração para compatibilidade com código existente
function normalize_config_format(config)
    normalized = Dict{String, Any}()
    
    # Handle nested structure from optimal_config.toml
    actual_config = haskey(config, "optimal_config") ? config["optimal_config"] : config
    
    # Extrai parâmetros PSO
    if haskey(actual_config, "pso")
        pso = actual_config["pso"]
        normalized["N"] = pso["N"]
        normalized["C1"] = pso["C1"]
        normalized["C2"] = pso["C2"]
        normalized["omega"] = pso["omega"]
        normalized["f_calls_limit"] = pso["f_calls_limit"]
        # Add bounds with defaults for backward compatibility
        normalized["lower_bounds"] = get(pso, "lower_bounds", [0.01, -0.25, -0.30, -0.20, 0.5, 2.0])
        normalized["upper_bounds"] = get(pso, "upper_bounds", [0.30, 0.25, 0.30, 0.20, 20.0, 50.0])
    else
        error("Missing required configuration section: pso")
    end
    
    # Extrai parâmetros de otimização
    if haskey(actual_config, "optimization")
        opt = actual_config["optimization"]
        normalized["use_lm"] = opt["use_lm"]
        normalized["temporal_penalty_weight"] = opt["temporal_penalty_weight"]
    else
        error("Missing required configuration section: optimization")
    end
    
    # Extrai parâmetros de detecção de outliers
    if haskey(actual_config, "outlier_detection")
        outlier = actual_config["outlier_detection"]
        normalized["mad_threshold"] = outlier["mad_threshold"]
        normalized["fator_liq"] = outlier["fator_liq"]
    else
        error("Missing required configuration section: outlier_detection")
    end

    # Extrai parâmetros de fit_curves (com defaults para retrocompatibilidade)
    fit_curves_config = get(actual_config, "fit_curves", Dict())
    normalized["continuity_search_days"] = get(fit_curves_config, "continuity_search_days", 30)
    normalized["min_bonds_for_fit"] = get(fit_curves_config, "min_bonds_for_fit", 3)
    normalized["lm_max_iterations"] = get(fit_curves_config, "lm_max_iterations", 50)
    normalized["reoptimization_cost_multiplier"] = get(fit_curves_config, "reoptimization_cost_multiplier", 10.0)
    normalized["reoptimization_pso_multiplier"] = get(fit_curves_config, "reoptimization_pso_multiplier", 2.0)
    
    # Add validation section if missing (for compatibility)
    if !haskey(actual_config, "validation")
        normalized["validation"] = Dict("num_hyperparameter_configs" => 5)
    end
    
    return normalized
end

# Gera datas úteis
function get_business_dates(start_date::Date, end_date::Date)
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

# Busca parâmetros de continuidade
function find_continuity_params(start_date::Date, config::Dict{String, Any}; log_level::Int=1)
    if log_level >= 2
        println("🔍 Buscando parâmetros de continuidade antes de $start_date...")
    end
    
    # Busca até X dias antes da data inicial, conforme configuração
    search_date = start_date - Day(1)
    
    for _ in 1:config["continuity_search_days"]
        if Dates.dayofweek(search_date) in 1:5  # Dia útil
            try
                if log_level >= 3
                    println("   Tentando $search_date...")
                end
                
                result = fit_nss_single_day(search_date, config, nothing; log_level=0) # Run silently
                if result.success
                    if log_level >= 2
                        println("✅ Parâmetros encontrados em $search_date")
                    end
                    return result.params
                end
            catch
                # Continua buscando
            end
        end
        search_date -= Day(1)
    end
    
    if log_level >= 2
        println("⚠️ Não foi possível encontrar parâmetros de continuidade")
    end
    return nothing
end

# Fit NSS para um dia
function fit_nss_single_day(date::Date, config::Dict{String, Any}, previous_params::Union{Vector{Float64}, Nothing}; log_level::Int=1, previous_cost::Union{Float64, Nothing}=nothing)
    try
        df = load_bacen_data(date, date)
        
        if nrow(df) < config["min_bonds_for_fit"]
            return DayResult(date, false, nothing, nothing, 0, 0, "Dados insuficientes (<$(config["min_bonds_for_fit"]) títulos)", false, false)
        end
        
        cash_flows, bond_quantities, _ = generate_cash_flows_with_quantity(df, date)
        
        if length(cash_flows) < config["min_bonds_for_fit"]
            return DayResult(date, false, nothing, nothing, 0, 0, "Cash flows insuficientes (<$(config["min_bonds_for_fit"]))", false, false)
        end
        
        if log_level >= 1
            print("📊 $date: $(length(cash_flows)) títulos")
        end
        
        # Otimização com remoção de outliers
        params, cost, final_cash_flows, outliers_removed, iterations = optimize_nelson_siegel_svensson_with_mad_outlier_removal(
            cash_flows, date, config["lower_bounds"], config["upper_bounds"];
            previous_params=previous_params,
            temporal_penalty_weight=config["temporal_penalty_weight"],
            pso_N=config["N"],
            pso_C1=config["C1"],
            pso_C2=config["C2"],
            pso_omega=config["omega"],
            pso_f_calls_limit=config["f_calls_limit"],
            fator_erro=config["mad_threshold"],
            fator_liq=config["fator_liq"],
            bond_quantities=bond_quantities,
            log_level=log_level
        )
        
        # Refinamento LM se configurado
        if config["use_lm"]
            try
                params_lm, cost_lm, lm_success = refine_nss_with_lbfgs(
                    final_cash_flows, date, params, config["lower_bounds"], config["upper_bounds"];
                    max_iterations=config["lm_max_iterations"], show_trace=false,
                    previous_params=previous_params,
                    temporal_penalty_weight=config["temporal_penalty_weight"],
                    log_level=log_level
                )
                
                if lm_success && cost_lm < cost
                    params = params_lm
                    cost = cost_lm
                    if log_level >= 2
                        print(" + LM")
                    end
                end
            catch
                # Mantém PSO se LM falhar
            end
        end
        
        used_previous = previous_params !== nothing
        
        # Tenta re-otimizar se o custo for muito alto em comparação com o dia anterior
        params, cost, final_cash_flows, outliers_removed, reoptimized = _reoptimize_if_needed(
            cost, previous_cost, config, cash_flows, date, params, final_cash_flows, outliers_removed, bond_quantities, previous_params, log_level
        )
        
        if log_level >= 1
            outlier_info = outliers_removed > 0 ? " (-$outliers_removed outliers)" : ""
            reopt_info = reoptimized ? " [RE-OPT]" : ""
            println(" → ✅ Custo: $(round(cost, digits=4))$outlier_info$reopt_info")
        end
        
        return DayResult(date, true, params, cost, length(final_cash_flows), outliers_removed, nothing, used_previous, reoptimized)
        
    catch e
        error_msg = string(e)
        if log_level >= 1
            println(" → ❌ $error_msg")
        end
        return DayResult(date, false, nothing, nothing, 0, 0, error_msg, false, false)
    end
end

function _reoptimize_if_needed(cost, previous_cost, config, cash_flows, date, params, final_cash_flows, outliers_removed, bond_quantities, previous_params, log_level)
    reoptimized = false
    reopt_multiplier = config["reoptimization_cost_multiplier"]

    if previous_cost === nothing || cost <= reopt_multiplier * previous_cost
        return params, cost, final_cash_flows, outliers_removed, reoptimized
    end

    if log_level >= 2
        println(" ⚠️ Custo elevado ($(round(cost, digits=4)) > $(reopt_multiplier)×$(round(previous_cost, digits=4))), re-otimizando...")
    end

    pso_multiplier = config["reoptimization_pso_multiplier"]
    params_reopt, cost_reopt, final_cash_flows_reopt, outliers_removed_reopt, _ = optimize_nelson_siegel_svensson_with_mad_outlier_removal(
        cash_flows, date, config["lower_bounds"], config["upper_bounds"];
        previous_params=previous_params,
        temporal_penalty_weight=config["temporal_penalty_weight"],
        pso_N=round(Int, config["N"] * pso_multiplier),
        pso_C1=config["C1"],
        pso_C2=config["C2"],
        pso_omega=config["omega"],
        pso_f_calls_limit=round(Int, config["f_calls_limit"] * pso_multiplier),
        fator_erro=config["mad_threshold"],
        fator_liq=config["fator_liq"],
        bond_quantities=bond_quantities,
        log_level=log_level
    )

    if config["use_lm"]
        try
            params_lm_reopt, cost_lm_reopt, lm_success_reopt = refine_nss_with_lbfgs(
                final_cash_flows_reopt, date, params_reopt, config["lower_bounds"], config["upper_bounds"];
                max_iterations=config["lm_max_iterations"], show_trace=false,
                previous_params=previous_params,
                temporal_penalty_weight=config["temporal_penalty_weight"],
                log_level=log_level
            )
            if lm_success_reopt && cost_lm_reopt < cost_reopt
                params_reopt = params_lm_reopt
                cost_reopt = cost_lm_reopt
                if log_level >= 2; print(" + LM-reopt"); end
            end
        catch
            # Mantém PSO re-otimizado se LM falhar
        end
    end

    if cost_reopt < cost
        if log_level >= 2; println(" 🚀 Re-otimização melhorou: $(round(cost_reopt, digits=4))"); end
        return params_reopt, cost_reopt, final_cash_flows_reopt, outliers_removed_reopt, true
    else
        if log_level >= 2; println(" 😞 Re-otimização não melhorou: $(round(cost_reopt, digits=4)) >= $(round(cost, digits=4))"); end
        return params, cost, final_cash_flows, outliers_removed, false
    end
end


# Fit sequencial principal
function fit_curves_sequential(start_date::Date, end_date::Date; 
                              find_continuity::Bool=false, log_level::Int=1, dry_run::Bool=false)
    
    if log_level >= 1
        println("📊 FIT SEQUENCIAL DE CURVAS NSS")
        println("🔗 COM CONTINUIDADE TEMPORAL")
        println("=" ^ 50)
    end
    
    # Carrega configuração
    config = read_optimal_config()
    
    if log_level >= 2
        println("✅ Configuração ótima:")
        println("   N=$(config["N"]), C1=$(config["C1"]), C2=$(config["C2"])")
        println("   ω=$(config["omega"]), LM=$(config["use_lm"])")
        println("   Temporal penalty=$(config["temporal_penalty_weight"])")
        println("   MAD threshold=$(config["mad_threshold"])")
        println("   Fator liquidez=$(config["fator_liq"]) ($(config["fator_liq"]*100)% do total)")
    end
    
    # Gera datas
    all_dates = get_business_dates(start_date, end_date)
    total_dates = length(all_dates)
    
    if log_level >= 1
        println("\\n📅 Período: $start_date a $end_date")
        println("📊 Datas úteis: $total_dates")
        println("⏱️ Estimativa: $(round(total_dates * 2 / 60, digits=1)) minutos")
    end
    
    if dry_run
        println("\\n🔍 DRY RUN - Configuração validada, não executando fits")
        return [], config
    end
    
    # Busca continuidade
    previous_params = nothing
    if find_continuity
        previous_params = find_continuity_params(start_date, config; log_level=log_level)
    end
    
    if log_level >= 1
        println("\\n🚀 Iniciando fit sequencial...")
    end
    
    # Processamento sequencial
    start_time = time()
    all_results = DayResult[]
    current_previous_params = previous_params
    current_previous_cost = nothing
    successful_fits = 0
    reoptimization_count = 0
    
    for (i, date) in enumerate(all_dates)
        if log_level >= 1 && (i % 50 == 1 || i == total_dates)
            progress_pct = round(i/total_dates*100, digits=1)
            println("🔄 Progresso: $i/$total_dates ($progress_pct%)")
        end
        
        result = fit_nss_single_day(date, config, current_previous_params; log_level=log_level, previous_cost=current_previous_cost)
        push!(all_results, result)
        
        if result.success
            current_previous_params = result.params
            current_previous_cost = result.cost
            successful_fits += 1
            
            if result.reoptimized
                reoptimization_count += 1
            end
        end
    end
    
    elapsed_time = time() - start_time
    
    if log_level >= 1
        println("\\n⏱️ Processamento completo em $(round(elapsed_time/60, digits=1)) minutos")
        
        continuity_count = sum(r.used_previous_params for r in all_results if r.success)
        println("🔗 Continuidade: $continuity_count/$successful_fits fits usaram previous_params")
        println("🚀 Re-otimizações: $reoptimization_count/$successful_fits fits foram re-otimizados por custo alto")
        println("📊 Taxa de sucesso: $(round(successful_fits/total_dates*100, digits=1))%")
    end
    
    return all_results, config
end

# Salva resultados
function save_results_to_csv(results::Vector{DayResult}, config::Dict{String, Any}, output_base::String)
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
        Reotimizado = Bool[],
        ErroMensagem = Union{String, Missing}[]
    )
    
    for result in results
        if result.success && result.params !== nothing
            push!(data, (
                result.date, result.success,
                result.params[1], result.params[2], result.params[3], 
                result.params[4], result.params[5], result.params[6],
                result.cost, result.n_bonds, result.outliers_removed,
                result.used_previous_params, result.reoptimized, missing
            ))
        else
            push!(data, (
                result.date, result.success,
                missing, missing, missing, missing, missing, missing,
                missing, result.n_bonds, result.outliers_removed,
                result.used_previous_params, result.reoptimized, result.error_message
            ))
        end
    end
    
    sort!(data, :Data)
    CSV.write(filename, data)
    
    # Estatísticas
    successful_fits = sum(data.Sucesso)
    total_fits = nrow(data)
    
    println("\\n💾 Dados salvos: $filename")
    println("📊 Resumo: $successful_fits/$total_fits sucessos ($(round(successful_fits/total_fits*100, digits=1))%)")
    
    if successful_fits > 0
        successful_data = filter(row -> row.Sucesso, data)
        continuity_rate = sum(successful_data.UsouPreviousParams) / successful_fits * 100
        reoptimization_rate = sum(successful_data.Reotimizado) / successful_fits * 100
        
        println("   Período: $(minimum(successful_data.Data)) a $(maximum(successful_data.Data))")
        println("   Continuidade: $(round(continuity_rate, digits=1))%")
        println("   Re-otimizações: $(round(reoptimization_rate, digits=1))%")
        println("   Custo médio: $(round(mean(skipmissing(successful_data.Custo)), digits=4))")
        println("   Títulos médios: $(round(mean(successful_data.NumTitulos), digits=1))")
    end
    
    return filename
end

# Função principal
function main()
    # --- CONFIGURAÇÃO DE LOG ---
    # Nível 1: Progresso (Padrão)
    # Nível 2: Informativo
    # Nível 3: Debug Completo
    log_level = 1
    # -------------------------

    args = get_default_args()
    
    # Parse datas
    try
        start_date = Date(args["start"])
        end_date = Date(args["end"])
        
        if start_date > end_date
            error("Data inicial não pode ser posterior à data final!")
        end
        
        if end_date > today()
            println("⚠️ Data final é no futuro, usando hoje como limite")
            end_date = today()
        end
        
        # Executa fit
        results, config = fit_curves_sequential(
            start_date, end_date;
            find_continuity=args["continuity"],
            log_level=log_level,
            dry_run=args["dry-run"]
        )
        
        if !args["dry-run"] && !isempty(results)
            filename = save_results_to_csv(results, config, args["output"])
            println("\\n🎉 Fit concluído! Arquivo: $filename")
        end
        
    catch e
        if isa(e, ArgumentError) && contains(string(e), "invalid Date")
            println("❌ Formato de data inválido. Use YYYY-MM-DD")
            exit(1)
        else
            println("❌ Erro: $e")
            exit(1)
        end
    end
end

# Executa se chamado diretamente
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end