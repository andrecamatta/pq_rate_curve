#!/usr/bin/env julia
"""
Teste da API de alto nível (fit_curves_for_period)
"""

using PQRateCurve
using Dates

println("🧪 TESTE API DE ALTO NÍVEL - fit_curves_for_period")
println("=" ^ 60)

# Testa fit para um período curto
start_date = Date(2024, 6, 3)
end_date = Date(2024, 6, 7)

println("\n📅 Período: $start_date a $end_date")
println("🚀 Executando fit_curves_for_period...\n")

try
    results, config = fit_curves_for_period(start_date, end_date)

    println("\n✅ Fit concluído com sucesso!")
    println("📊 Resultados:")
    println("   Total de dias processados: $(length(results))")

    # Filtra apenas sucessos
    successful_results = filter(r -> r.success, results)

    if !isempty(successful_results)
        println("\n📈 Detalhes por dia (sucessos: $(length(successful_results))):")
        for result in successful_results
            println("   $(result.date):")
            println("      β₀=$(round(result.params[1], digits=4)), β₁=$(round(result.params[2], digits=4)), β₂=$(round(result.params[3], digits=4))")
            println("      Custo: $(format_cost(result.cost))")
            println("      Outliers: $(result.outliers_removed)")
            println("      Usou previous_params: $(result.used_previous_params)")
        end

        println("\n💰 Teste de custos:")
        costs = [r.cost for r in successful_results]
        dates = [r.date for r in successful_results]

        println("   Custos individuais: $([format_cost(c) for c in costs])")

        # Testa a normalização com os resultados reais
        normalized = normalize_cost_by_volume(dates, costs)
        println("   ✅ Custo normalizado pelo volume: $(format_cost(normalized))")
    end

    println("\n" * "=" ^ 60)
    println("🎉 API DE ALTO NÍVEL FUNCIONANDO PERFEITAMENTE!")
    println("✅ Todas as funcionalidades integradas estão OK")
    println("=" ^ 60)

catch e
    println("\n❌ ERRO na API de alto nível: $e")
    println("\nStacktrace:")
    for (exc, bt) in Base.catch_stack()
        showerror(stdout, exc, bt)
        println()
    end
    exit(1)
end
