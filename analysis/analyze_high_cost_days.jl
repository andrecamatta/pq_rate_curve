#!/usr/bin/env julia
"""
Analisa dias problemáticos com custo acima de 1000
"""

using PQRateCurve
using DataFrames, Dates, Statistics

println("🔍 ANÁLISE DE DIAS PROBLEMÁTICOS - CUSTO > 1000")
println("=" ^ 70)

# Abre banco histórico
println("📂 Abrindo banco de dados histórico...")
db = init_database("historical_curves.db")

# Obter estatísticas
stats = get_database_stats(db)
println("✅ Banco aberto com sucesso!")
println("   📊 Total de curvas: $(stats.total_curves)")
println("   ✅ Bem-sucedidas: $(stats.successful_curves) ($(stats.success_rate)%)")
println("   📅 Período: $(stats.date_range[1]) → $(stats.date_range[2])")
println()

# Carrega todas as curvas
println("📥 Carregando todas as curvas...")
df_curves = load_curves(db, stats.date_range[1], stats.date_range[2])
println("✅ Carregadas $(nrow(df_curves)) curvas")
println()

# Filtra curvas bem-sucedidas com custo > 1000
println("🔎 Filtrando curvas com custo > 1000...")
df_problematic = df_curves[
    (df_curves.success .== 1) .&
    (.!ismissing.(df_curves.cost)) .&
    (df_curves.cost .> 1000.0),
    :]

sort!(df_problematic, :cost, rev=true)

println("=" ^ 70)
println("📊 DIAS PROBLEMÁTICOS COM CUSTO > 1000")
println("=" ^ 70)
println()

if nrow(df_problematic) == 0
    println("✅ Nenhum dia com custo acima de 1000 encontrado!")
    println("   Todas as curvas bem-sucedidas tiveram custo ≤ 1000")
else
    println("⚠️  Total de dias problemáticos: $(nrow(df_problematic))")
    println("   Representam $(round(nrow(df_problematic)/stats.successful_curves*100, digits=2))% das curvas bem-sucedidas")
    println()

    # Estatísticas
    println("📈 ESTATÍSTICAS DOS ERROS:")
    println("   Maior custo: $(round(maximum(df_problematic.cost), digits=2))")
    println("   Menor custo (>1000): $(round(minimum(df_problematic.cost), digits=2))")
    println("   Custo médio: $(round(mean(df_problematic.cost), digits=2))")
    println("   Mediana: $(round(median(df_problematic.cost), digits=2))")
    println()

    # Distribuição por ano
    println("📅 DISTRIBUIÇÃO POR ANO:")
    df_problematic.year = year.(df_problematic.date)
    year_counts = combine(groupby(df_problematic, :year), nrow => :count)
    sort!(year_counts, :year)
    for row in eachrow(year_counts)
        println("   $(row.year): $(row.count) dias")
    end
    println()

    # Top 20 piores dias
    println("🔴 TOP 20 PIORES DIAS:")
    println("=" ^ 70)
    println("Ranking | Data       | Custo      | Títulos | Outliers | Params")
    println("-" ^ 70)

    for (i, row) in enumerate(eachrow(df_problematic[1:min(20, nrow(df_problematic)), :]))
        params_str = "β₀=$(round(row.beta0, digits=2))"
        println("$(lpad(i, 7)) | $(row.date) | $(lpad(round(row.cost, digits=2), 10)) | $(lpad(row.n_bonds, 7)) | $(lpad(row.outliers_removed, 8)) | $params_str")
    end
    println("=" ^ 70)

    # Lista COMPLETA de todos os dias problemáticos
    println()
    println("📋 LISTA COMPLETA DE TODOS OS DIAS COM CUSTO > 1000:")
    println("=" ^ 70)
    for (i, row) in enumerate(eachrow(df_problematic))
        println("$(lpad(i, 3)). $(row.date) | Custo: $(lpad(round(row.cost, digits=2), 10)) | Títulos: $(row.n_bonds) | Outliers: $(row.outliers_removed)")
    end
    println("=" ^ 70)
end

println()
println("✅ ANÁLISE COMPLETA!")
