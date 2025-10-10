#!/usr/bin/env julia
"""
Analisa o impacto de aumentar min_bonds_for_fit de 3 para 6
"""

using PQRateCurve
using DataFrames, Dates, Statistics

println("🔍 ANÁLISE DE IMPACTO: min_bonds_for_fit 3 → 6")
println("=" ^ 70)

# Abre banco histórico
println("📂 Abrindo banco de dados histórico...")
db = init_database("historical_curves.db")

# Obter estatísticas
stats = get_database_stats(db)
println("✅ Banco aberto com sucesso!")
println("   📊 Total de curvas: $(stats.total_curves)")
println("   ✅ Bem-sucedidas: $(stats.successful_curves)")
println("   ❌ Com falha: $(stats.failed_curves)")
println()

# Carrega todas as curvas
println("📥 Carregando todas as curvas...")
df_curves = load_curves(db, stats.date_range[1], stats.date_range[2])
println("✅ Carregadas $(nrow(df_curves)) curvas")
println()

# Analisa quantos dias seriam afetados
println("🔎 Analisando impacto da mudança...")
println()

# Dias que eram bem-sucedidos com 3 bonds mas teriam falhado com 6
# (após remover outliers)
df_success = df_curves[df_curves.success .== 1, :]

# Calcula títulos efetivos após outliers
df_success.effective_bonds = df_success.n_bonds .- df_success.outliers_removed

# Filtra os que teriam falhado com min=6
df_would_fail = df_success[df_success.effective_bonds .< 6, :]
sort!(df_would_fail, :effective_bonds)

println("=" ^ 70)
println("📊 IMPACTO DA MUDANÇA: min_bonds_for_fit 3 → 6")
println("=" ^ 70)
println()

println("📈 ESTATÍSTICAS:")
println("   Total de dias bem-sucedidos: $(nrow(df_success))")
println("   Dias afetados pela mudança: $(nrow(df_would_fail))")
println("   Percentual afetado: $(round(nrow(df_would_fail)/nrow(df_success)*100, digits=2))%")
println()

if nrow(df_would_fail) > 0
    println("📉 DISTRIBUIÇÃO DE TÍTULOS EFETIVOS (dias afetados):")
    println("   Mínimo: $(minimum(df_would_fail.effective_bonds)) títulos")
    println("   Máximo: $(maximum(df_would_fail.effective_bonds)) títulos")
    println("   Média: $(round(mean(df_would_fail.effective_bonds), digits=1)) títulos")
    println()

    # Distribuição por número de títulos efetivos
    println("📊 DISTRIBUIÇÃO:")
    for n in sort(unique(df_would_fail.effective_bonds))
        count = sum(df_would_fail.effective_bonds .== n)
        println("   $n títulos: $count dias")
    end
    println()

    # Distribuição por ano
    println("📅 DISTRIBUIÇÃO POR ANO:")
    df_would_fail.year = year.(df_would_fail.date)
    year_counts = combine(groupby(df_would_fail, :year), nrow => :count)
    sort!(year_counts, :year)
    for row in eachrow(year_counts)
        pct = round(row.count / nrow(df_would_fail) * 100, digits=1)
        println("   $(row.year): $(row.count) dias ($(pct)%)")
    end
    println()

    # Top 20 dias afetados com maior custo
    println("🔴 TOP 20 DIAS AFETADOS (ordenados por custo):")
    println("=" ^ 70)
    df_sorted = sort(df_would_fail, :cost, rev=true)
    println("Ranking | Data       | Custo      | Títulos | Outliers | Efetivos")
    println("-" ^ 70)
    for (i, row) in enumerate(eachrow(df_sorted[1:min(20, nrow(df_sorted)), :]))
        println("$(lpad(i, 7)) | $(row.date) | $(lpad(round(row.cost, digits=2), 10)) | $(lpad(row.n_bonds, 7)) | $(lpad(row.outliers_removed, 8)) | $(lpad(row.effective_bonds, 8))")
    end
    println("=" ^ 70)
    println()

    # Lista completa de dias afetados
    println("📋 LISTA COMPLETA DE DIAS QUE PASSARIAM A FALHAR:")
    println("=" ^ 70)
    for (i, row) in enumerate(eachrow(df_would_fail))
        println("$(lpad(i, 3)). $(row.date) | Efetivos: $(row.effective_bonds) | Custo: $(lpad(round(row.cost, digits=2), 10)) | Total: $(row.n_bonds) | Removidos: $(row.outliers_removed)")
    end
    println("=" ^ 70)
else
    println("✅ NENHUM DIA SERIA AFETADO!")
    println("   Todos os dias bem-sucedidos tinham ≥ 6 títulos efetivos.")
end

println()
println("💡 RECOMENDAÇÃO:")
if nrow(df_would_fail) == 0
    println("   ✅ Mudança segura! Nenhum impacto no histórico.")
elseif nrow(df_would_fail) / nrow(df_success) < 0.01
    println("   ✅ Mudança recomendada! Impacto mínimo (< 1%).")
    println("   Os dias afetados provavelmente têm fits de baixa qualidade.")
elseif nrow(df_would_fail) / nrow(df_success) < 0.05
    println("   ⚠️  Mudança aceitável. Impacto moderado (< 5%).")
    println("   Revisar dias afetados para confirmar que são problemáticos.")
else
    println("   ⚠️  Impacto significativo (≥ 5%). Considerar:")
    println("   - Revisar critério de outlier detection")
    println("   - Avaliar se min_bonds_for_fit = 6 é muito restritivo")
    println("   - Implementar interpolação para dias com poucos títulos")
end

println()
println("✅ ANÁLISE COMPLETA!")
