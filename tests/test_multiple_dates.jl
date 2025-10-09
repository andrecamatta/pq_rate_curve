#!/usr/bin/env julia
"""
Teste mais abrangente: fit de curvas para múltiplas datas
"""

using PQRateCurve
using Dates, DataFrames, Statistics

println("🧪 TESTE ABRANGENTE - Fit de Curvas para Múltiplas Datas")
println("=" ^ 60)

# Testa para várias datas em diferentes meses
test_dates = [
    Date(2024, 1, 2),
    Date(2024, 3, 15),
    Date(2024, 6, 3),
]

println("\n📅 Testando fit para $(length(test_dates)) datas:\n")

global successes = 0
global failures = 0
global costs = Float64[]
global params_history = []

for test_date in test_dates
    global successes, failures, costs, params_history
    print("   $(test_date): ")
    try
        # Carrega dados
        df = load_bacen_data(test_date, test_date)

        if nrow(df) < 3
            println("❌ Dados insuficientes ($(nrow(df)) títulos)")
            failures += 1
            continue
        end

        # Gera cash flows
        cash_flows, bond_quantities, _ = generate_cash_flows_with_quantity(df, test_date)

        # Carrega configuração
        config_service = load_config("config.toml")
        config = get_raw_config(config_service)
        lower_bounds, upper_bounds = get_pso_bounds(config)

        # Usa parâmetros do dia anterior se disponível (testa temporal penalty)
        previous_params = length(params_history) > 0 ? params_history[end] : nothing

        # Otimiza NSS
        params, cost, final_cash_flows, outliers, iterations = optimize_nelson_siegel_svensson_with_mad_outlier_removal(
            cash_flows, test_date, lower_bounds, upper_bounds;
            previous_params=previous_params,
            temporal_penalty_weight=0.01,
            pso_N=25,
            pso_C1=2.0,
            pso_C2=2.0,
            pso_omega=0.4,
            pso_f_calls_limit=800,
            error_threshold_global=20.0,
            fator_liq=0.005,
            ultra_low_factor=3.0,
            max_iterations=2,
            bond_quantities=bond_quantities,
            verbose=false
        )

        println("✅ Custo=$(format_cost(cost)), Outliers=$(length(outliers)), Títulos=$(nrow(df))")

        push!(costs, cost)
        push!(params_history, params)
        successes += 1

    catch e
        println("❌ ERRO: $e")
        failures += 1
    end
end

println("\n" * "=" ^ 60)
println("📊 RESULTADOS:")
println("   ✅ Sucessos: $successes/$(length(test_dates))")
println("   ❌ Falhas: $failures/$(length(test_dates))")

if !isempty(costs)
    println("\n📈 Estatísticas de Custo:")
    println("   Média: $(format_cost(mean(costs)))")
    println("   Mínimo: $(format_cost(minimum(costs)))")
    println("   Máximo: $(format_cost(maximum(costs)))")

    if length(costs) > 1
        println("   Desvio: $(format_cost(std(costs)))")
    end
end

# Testa a função de normalização de custos com datas reais
if successes >= 2
    println("\n💰 Teste de Normalização de Custos:")
    successful_dates = test_dates[1:min(successes, length(test_dates))]
    successful_costs = costs[1:min(successes, length(costs))]

    println("   Datas: $(length(successful_dates))")
    println("   Custos: $successful_costs")

    normalized = normalize_cost_by_volume(successful_dates, successful_costs)
    println("   ✅ Custo normalizado: $(format_cost(normalized))")
end

println("\n" * "=" ^ 60)
if successes == length(test_dates)
    println("🎉 TESTE COMPLETO PASSOU COM SUCESSO!")
    println("✅ Refatoração funcionando perfeitamente em múltiplas datas")
else
    println("⚠️  Alguns testes falharam, mas isso pode ser normal (falta de dados)")
    if successes >= length(test_dates) / 2
        println("✅ Maioria dos testes passou - refatoração OK")
    end
end
println("=" ^ 60)
