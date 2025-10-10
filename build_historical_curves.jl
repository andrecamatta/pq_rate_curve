#!/usr/bin/env julia
"""
Script para construir base histórica completa de curvas NSS
Período: Fevereiro/2015 → Setembro/2025
"""

using PQRateCurve
using Dates

println("=" ^ 70)
println("🏗️  CONSTRUÇÃO DE BASE HISTÓRICA DE CURVAS NSS")
println("=" ^ 70)

# Configuração
DB_PATH = "historical_curves.db"
START_DATE = Date(2015, 2, 1)   # Primeira data viável (90% sucesso)
END_DATE = Date(2025, 9, 30)    # Última data disponível
CONFIG_FILE = "config.toml"

println("\n📋 Configuração:")
println("   Banco de dados: $DB_PATH")
println("   Período: $START_DATE → $END_DATE")
println("   Config: $CONFIG_FILE")

# Verifica se banco já existe
if isfile(DB_PATH)
    println("\n💾 Banco de dados existente encontrado")
    println("   Modo: Processamento incremental (só datas faltantes)")

    db = init_database(DB_PATH)
    stats = get_database_stats(db)

    if stats.total_curves > 0
        println("\n📊 Estatísticas atuais:")
        println("   Total de curvas: $(stats.total_curves)")
        println("   Bem-sucedidas: $(stats.successful_curves)")
        println("   Taxa de sucesso: $(stats.success_rate)%")
        println("   Período: $(stats.date_range[1]) → $(stats.date_range[2])")
    end
else
    println("\n🆕 Criando novo banco de dados")

    # Estima tempo
    total_dates = length(get_business_dates(START_DATE, END_DATE))
    println("\n📊 Estimativas:")
    println("   Total de dias úteis: ~$total_dates")
    println("   Tempo estimado: ~$(round(total_dates * 2 / 60, digits=1)) minutos")
end

println("\n" * "=" ^ 70)
println("🚀 Iniciando processamento...")
println("=" ^ 70)

# Executa fit com persistência SQLite
results, config = fit_curves_for_period(
    START_DATE,
    END_DATE;
    config_file=CONFIG_FILE,
    db_path=DB_PATH,
    output_csv=nothing,  # Não gera CSV (dados no banco)
    find_continuity=true,
    verbose=true
)

# Estatísticas finais
println("\n" * "=" ^ 70)
println("📊 PROCESSAMENTO CONCLUÍDO")
println("=" ^ 70)

db = init_database(DB_PATH)
final_stats = get_database_stats(db)

println("\n📈 Estatísticas finais:")
println("   Total de curvas: $(final_stats.total_curves)")
println("   Bem-sucedidas: $(final_stats.successful_curves)")
println("   Falhadas: $(final_stats.failed_curves)")
println("   Taxa de sucesso: $(final_stats.success_rate)%")
println("   Período: $(final_stats.date_range[1]) → $(final_stats.date_range[2])")

# Análise dos resultados desta execução
successful = filter(r -> r.success, results)
if !isempty(successful)
    using Statistics

    costs = [r.cost for r in successful]
    beta0_values = [r.params[1] for r in successful]

    println("\n📊 Análise dos parâmetros:")
    println("   β₀ médio: $(round(mean(beta0_values), digits=4))")
    println("   β₀ range: $(round(minimum(beta0_values), digits=4)) - $(round(maximum(beta0_values), digits=4))")
    println("   Custo médio: R\$ $(round(mean(costs), digits=2))")
    println("   Custo mediano: R\$ $(round(median(costs), digits=2))")
end

println("\n" * "=" ^ 70)
println("✅ Base histórica salva em: $DB_PATH")
println("=" ^ 70)
