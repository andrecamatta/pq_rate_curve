#!/usr/bin/env julia
"""
build_historical_database.jl

Script de exemplo: Construção de base histórica de curvas NSS com SQLite

Demonstra como usar o sistema de persistência SQLite para:
1. Processar curvas NSS para um período extenso (2015-2025)
2. Processamento incremental (rodar novamente só processa datas novas)
3. Retomar execuções interrompidas automaticamente
4. Consultar estatísticas do banco

Uso:
    julia --project=. examples/build_historical_database.jl
"""

using PQRateCurve
using Dates

println("=" ^ 70)
println("🏗️  CONSTRUÇÃO DE BASE HISTÓRICA DE CURVAS NSS")
println("=" ^ 70)

# Configuração
DB_PATH = "curves.db"
START_DATE = Date(2015, 2, 1)   # Primeira data viável (90% sucesso)
END_DATE = Date(2025, 9, 30)    # Última data com dados disponíveis

CONFIG_FILE = "config.toml"

println("\n📋 Configuração:")
println("   Banco de dados: $DB_PATH")
println("   Período: $START_DATE → $END_DATE")
println("   Config: $CONFIG_FILE")

# Verifica se banco já existe
if isfile(DB_PATH)
    println("\n💾 Banco de dados existente encontrado")
    println("   Modo: Processamento incremental (só datas faltantes)")

    # Mostra estatísticas atuais
    db = init_database(DB_PATH)
    stats = get_database_stats(db)

    if stats.total_curves > 0
        println("\n📊 Estatísticas do banco:")
        println("   Total de curvas: $(stats.total_curves)")
        println("   Bem-sucedidas: $(stats.successful_curves)")
        println("   Falhadas: $(stats.failed_curves)")
        println("   Taxa de sucesso: $(stats.success_rate)%")
        println("   Período: $(stats.date_range[1]) → $(stats.date_range[2])")

        # Calcula quantas faltam
        total_expected = length(get_business_dates(START_DATE, END_DATE))
        remaining = total_expected - stats.total_curves

        if remaining > 0
            println("\n⏳ Faltam processar: ~$remaining dias úteis")
            println("   Tempo estimado: ~$(round(remaining * 2 / 60, digits=1)) minutos")
        else
            println("\n✅ Base histórica completa!")
        end
    end
else
    println("\n🆕 Banco de dados será criado")
    println("   Modo: Primeira execução (processará todas as datas)")

    total_dates = length(get_business_dates(START_DATE, END_DATE))
    println("\n📊 Estimativas:")
    println("   Total de dias úteis: ~$total_dates")
    println("   Tempo estimado: ~$(round(total_dates * 2 / 60, digits=1)) minutos")
    println("   Tamanho final esperado: ~$(round(total_dates * 0.5 / 1024, digits=1)) MB")
end

println("\n" * "=" ^ 70)
println("🚀 Iniciando processamento...")
println("=" ^ 70)

# Executa o fit com banco de dados
# O sistema automaticamente:
# - Cria o banco se não existir
# - Carrega datas já processadas
# - Processa apenas datas faltantes
# - Salva cada resultado no banco
# - Pode ser interrompido e retomado a qualquer momento

results, config = fit_curves_for_period(
    START_DATE,
    END_DATE;
    config_file=CONFIG_FILE,
    db_path=DB_PATH,
    output_csv=nothing,  # Não gera CSV (dados já no banco)
    find_continuity=true,
    verbose=true
)

# Estatísticas finais
println("\n" * "=" ^ 70)
println("📊 PROCESSAMENTO CONCLUÍDO")
println("=" ^ 70)

# Recarrega estatísticas finais
db = init_database(DB_PATH)
final_stats = get_database_stats(db)

println("\n📈 Estatísticas finais:")
println("   Total de curvas no banco: $(final_stats.total_curves)")
println("   Bem-sucedidas: $(final_stats.successful_curves)")
println("   Falhadas: $(final_stats.failed_curves)")
println("   Taxa de sucesso: $(final_stats.success_rate)%")
println("   Período coberto: $(final_stats.date_range[1]) → $(final_stats.date_range[2])")

# Análise dos parâmetros
successful = filter(r -> r.success, results)
if !isempty(successful)
    # Médias dos parâmetros
    beta0_values = [r.params[1] for r in successful]
    tau1_values = [r.params[5] for r in successful]
    tau2_values = [r.params[6] for r in successful]
    costs = [r.cost for r in successful]

    println("\n📊 Análise dos parâmetros:")
    println("   β₀ médio: $(round(mean(beta0_values), digits=4)) (range: $(round(minimum(beta0_values), digits=4))-$(round(maximum(beta0_values), digits=4)))")
    println("   τ₁ médio: $(round(mean(tau1_values), digits=2)) anos")
    println("   τ₂ médio: $(round(mean(tau2_values), digits=2)) anos")
    println("   Custo médio: $(format_cost(mean(costs)))")
    println("   Custo mediano: $(format_cost(median(costs)))")
end

println("\n💡 Próximos passos:")
println("   1. Consultar dados: curves = load_curves(db, Date(2024,1,1), Date(2024,12,31))")
println("   2. Análises SQL: Conecte-se ao banco com qualquer ferramenta SQLite")
println("   3. Incrementar: Execute este script novamente para adicionar datas novas")
println("   4. Exportar CSV: Use output_csv=\"historico\" na função fit_curves_for_period()")

println("\n" * "=" ^ 70)
println("✅ Concluído! Banco salvo em: $DB_PATH")
println("=" ^ 70)
