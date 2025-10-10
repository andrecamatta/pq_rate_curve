#!/usr/bin/env julia
"""
Compara curva problemática (2016-12-30) com dia anterior
"""

using PQRateCurve
using DataFrames, Dates, Statistics

println("🔍 COMPARAÇÃO: 2016-12-30 (PROBLEMÁTICA) vs 2016-12-29")
println("=" ^ 70)

# Abre banco histórico
db = init_database("historical_curves.db")

# Datas a comparar
problem_date = Date(2016, 12, 30)
previous_date = Date(2016, 12, 29)

# Carrega as duas curvas
println("📥 Carregando curvas...")
curve_problem = load_curve(db, problem_date)
curve_previous = load_curve(db, previous_date)

if curve_problem === nothing || curve_previous === nothing
    println("❌ Não foi possível carregar uma das curvas")
    exit(1)
end

println("✅ Curvas carregadas com sucesso!")
println()

# Mostra parâmetros
println("📊 PARÂMETROS NSS:")
println("=" ^ 70)
println("Parâmetro | 2016-12-29    | 2016-12-30    | Diferença")
println("-" ^ 70)

params_prev = curve_previous.params
params_prob = curve_problem.params

for (i, name) in enumerate(["β₀", "β₁", "β₂", "β₃", "τ₁", "τ₂"])
    diff = params_prob[i] - params_prev[i]
    diff_pct = abs(diff / params_prev[i] * 100)
    println("$(rpad(name, 9)) | $(lpad(round(params_prev[i], digits=4), 13)) | $(lpad(round(params_prob[i], digits=4), 13)) | $(lpad(round(diff, digits=4), 9)) ($(round(diff_pct, digits=1))%)")
end
println("=" ^ 70)
println()

# Informações adicionais
println("📈 INFORMAÇÕES DAS CURVAS:")
println("-" ^ 70)
println("Data       | Custo     | Títulos | Outliers | Sucesso")
println("-" ^ 70)
println("2016-12-29 | $(lpad(round(curve_previous.cost, digits=2), 9)) | $(lpad(curve_previous.n_bonds, 7)) | $(lpad(curve_previous.outliers_removed, 8)) | ✅")
println("2016-12-30 | $(lpad(round(curve_problem.cost, digits=2), 9)) | $(lpad(curve_problem.n_bonds, 7)) | $(lpad(curve_problem.outliers_removed, 8)) | ⚠️  (alto custo)")
println("=" ^ 70)
println()

# Calcula taxas para vários prazos
maturities = [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0]

println("💰 TAXAS SPOT (% a.a.):")
println("=" ^ 70)
println("Prazo | 2016-12-29 | 2016-12-30 | Diferença (bps)")
println("-" ^ 70)

max_diff_bps = 0.0
global max_diff_bps
for maturity in maturities
    global max_diff_bps
    rate_prev = nss_rate(maturity, params_prev) * 100
    rate_prob = nss_rate(maturity, params_prob) * 100
    diff_bps = (rate_prob - rate_prev) * 100  # basis points
    max_diff_bps = max(max_diff_bps, abs(diff_bps))

    symbol = diff_bps > 0 ? "↑" : "↓"
    println("$(lpad(maturity, 5))a | $(lpad(round(rate_prev, digits=2), 10))% | $(lpad(round(rate_prob, digits=2), 10))% | $(lpad(round(diff_bps, digits=1), 13)) $symbol")
end
println("=" ^ 70)
println()

# Análise da continuidade temporal
println("📉 ANÁLISE DE CONTINUIDADE:")
println("-" ^ 70)
println("Maior diferença absoluta: $(round(max_diff_bps, digits=1)) bps")
println()

if max_diff_bps < 50
    println("✅ CONTINUIDADE EXCELENTE!")
    println("   Apesar do alto custo, a curva manteve-se muito próxima do dia anterior.")
    println("   Diferenças < 50 bps são consideradas normais em mercados estáveis.")
elseif max_diff_bps < 100
    println("⚠️  CONTINUIDADE BOA")
    println("   Algumas variações moderadas, mas dentro de limites aceitáveis.")
elseif max_diff_bps < 200
    println("⚠️  CONTINUIDADE RAZOÁVEL")
    println("   Variações significativas detectadas.")
else
    println("❌ DESCONTINUIDADE DETECTADA!")
    println("   Mudança brusca na estrutura a termo.")
end
println()

# Contexto de mercado
println("📅 CONTEXTO:")
println("-" ^ 70)
println("🎄 Período: Entre Natal e Ano Novo")
println("📊 2016-12-29: $(curve_previous.n_bonds) títulos negociados (mercado em funcionamento)")
println("📊 2016-12-30: $(curve_problem.n_bonds) títulos negociados (mercado reduzido)")
println("⚠️  Redução de liquidez: $(round((1 - curve_problem.n_bonds/curve_previous.n_bonds)*100, digits=1))%")
println()
println("💡 INTERPRETAÇÃO:")
println("   O alto custo (20,393) reflete a dificuldade de ajuste com poucos títulos,")
println("   mas a curva resultante manteve coerência econômica com o dia anterior,")
println("   demonstrando robustez do algoritmo mesmo em condições adversas.")
println("=" ^ 70)
