#!/usr/bin/env julia
"""
Script de teste rápido para verificar se a refatoração não quebrou nada
"""

using PQRateCurve
using Dates, DataFrames

println("🧪 TESTE DE REFATORAÇÃO - Verificando funcionalidades")
println("=" ^ 60)

# Teste 1: Funções de formatação
println("\n✅ Teste 1: Funções de Formatação")
println("-" ^ 60)
try
    @assert format_percentage(0.1234) == "12.34%"
    @assert format_score(1.123456789) == 1.123457
    @assert format_basis_points(12.3456) == 12.3
    @assert format_currency(1234.567) == 1234.57

    # Testa format_nss_params
    params = [0.123456, -0.054321, 0.012345, -0.006789, 5.123456, 12.654321]
    formatted = format_nss_params(params)
    @assert length(formatted) == 6
    @assert formatted[1] == 0.1235  # β₀
    @assert formatted[5] == 5.12    # τ₁
    @assert formatted[6] == 12.65   # τ₂

    println("   ✅ format_percentage: $(format_percentage(0.1234))")
    println("   ✅ format_score: $(format_score(1.123456789))")
    println("   ✅ format_basis_points: $(format_basis_points(12.3456))")
    println("   ✅ format_currency: $(format_currency(1234.567))")
    println("   ✅ format_nss_params: OK (6 parâmetros formatados)")
    println("   ✅ Todas as funções de formatação funcionando!")
catch e
    println("   ❌ ERRO nas funções de formatação: $e")
    exit(1)
end

# Teste 2: Fit de curva para uma data específica
println("\n✅ Teste 2: Fit de Curva (NSS com PSO)")
println("-" ^ 60)
try
    test_date = Date(2024, 6, 3)  # Data de exemplo
    println("   📅 Testando fit para: $test_date")

    # Carrega dados
    df = load_bacen_data(test_date, test_date)
    println("   📊 Dados carregados: $(nrow(df)) títulos")

    if nrow(df) >= 3
        # Gera cash flows
        cash_flows, bond_quantities, _ = generate_cash_flows_with_quantity(df, test_date)
        println("   💰 Cash flows gerados: $(length(cash_flows)) fluxos")

        # Carrega configuração
        config_service = load_config("config.toml")
        config = get_raw_config(config_service)
        lower_bounds, upper_bounds = get_pso_bounds(config)

        # Otimiza NSS
        params, cost, final_cash_flows, outliers, iterations = fit_nss(
            cash_flows, test_date, lower_bounds, upper_bounds;
            pso_N=30,
            pso_C1=2.0,
            pso_C2=2.0,
            pso_omega=0.4,
            pso_f_calls_limit=1000,
            error_threshold_global=20.0,
            fator_liq=0.005,
            ultra_low_factor=3.0,
            max_iterations=2,
            bond_quantities=bond_quantities,
            verbose=false
        )

        println("   ✅ Otimização concluída!")
        println("   📈 Parâmetros NSS: β₀=$(round(params[1], digits=4)), β₁=$(round(params[2], digits=4)), β₂=$(round(params[3], digits=4)), β₃=$(round(params[4], digits=4)), τ₁=$(round(params[5], digits=2)), τ₂=$(round(params[6], digits=2))")
        println("   💲 Custo: $(format_cost(cost))")
        println("   🎯 Outliers detectados: $(length(outliers))")
        println("   🔄 Iterações: $iterations")

        # Testa taxa NSS para alguns vértices
        vertices = [0.5, 1.0, 5.0, 10.0]
        println("   📊 Taxas NSS nos vértices:")
        for v in vertices
            rate = nss_rate(v, params)
            println("      $(v) ano(s): $(format_percentage(rate))")
        end

    else
        println("   ⚠️  Dados insuficientes para teste ($nrow(df) < 3)")
    end

catch e
    println("   ❌ ERRO no fit de curva: $e")
    println("   Stacktrace:")
    for (exc, bt) in Base.catch_stack()
        showerror(stdout, exc, bt)
        println()
    end
    exit(1)
end

# Teste 3: Função de normalização de custos
println("\n✅ Teste 3: Normalização de Custos por Volume")
println("-" ^ 60)
try
    # Testa com algumas datas
    test_dates = [Date(2024, 6, 3), Date(2024, 6, 4), Date(2024, 6, 5)]
    test_costs = [100.0, 150.0, 200.0]

    println("   📅 Datas: $(length(test_dates))")
    println("   💰 Custos: $test_costs")

    normalized_cost = normalize_cost_by_volume(test_dates, test_costs)

    println("   ✅ Custo normalizado: $(format_cost(normalized_cost))")
    println("   ✅ Função normalize_cost_by_volume funcionando!")

catch e
    println("   ❌ ERRO na normalização: $e")
    println("   Stacktrace:")
    for (exc, bt) in Base.catch_stack()
        showerror(stdout, exc, bt)
        println()
    end
    exit(1)
end

println("\n" * "=" ^ 60)
println("🎉 TODOS OS TESTES PASSARAM COM SUCESSO!")
println("✅ Refatoração não quebrou nenhuma funcionalidade")
println("=" ^ 60)
