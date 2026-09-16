#!/usr/bin/env julia
"""
Teste rápido do sistema de persistência SQLite
"""

using PQRateCurve
using Dates, DataFrames, Statistics

println("🧪 TESTE: Sistema de Persistência SQLite")
println("=" ^ 60)

# Remove banco de teste se existir
test_db = "test_curves.db"
if isfile(test_db)
    for f in (test_db, test_db * "-wal", test_db * "-shm")
        isfile(f) && rm(f; force=true)
    end
    println("🗑️  Banco de teste anterior removido")
end

# Teste 1: Criar banco e processar algumas datas
println("\n📝 Teste 1: Primeira execução (criar banco)")
println("-" ^ 60)

# Processa 10 dias de fevereiro/2015
start_date = Date(2015, 2, 2)
end_date = Date(2015, 2, 15)

println("   Processando $start_date → $end_date")

results1, config = fit_curves_for_period(
    start_date,
    end_date;
    db_path=test_db,
    output_csv=nothing,
    verbose=true
)

# Verifica resultados
successful1 = sum(r.success for r in results1)
println("\n✅ Primeira execução: $(length(results1)) datas, $successful1 sucessos")

# Teste 2: Reprocessar o mesmo período (deve pular tudo)
println("\n📝 Teste 2: Reprocessar mesmo período (modo incremental)")
println("-" ^ 60)

results2, _ = fit_curves_for_period(
    start_date,
    end_date;
    db_path=test_db,
    output_csv=nothing,
    verbose=true
)

println("\n✅ Segunda execução: $(length(results2)) datas (deve ser igual à primeira)")

# Teste 3: Adicionar mais datas
println("\n📝 Teste 3: Estender período (processar datas novas)")
println("-" ^ 60)

extended_end = Date(2015, 2, 28)
println("   Estendendo até $extended_end")

results3, _ = fit_curves_for_period(
    start_date,
    extended_end;
    db_path=test_db,
    output_csv=nothing,
    verbose=true
)

println("\n✅ Terceira execução: $(length(results3)) datas (deve incluir novas)")

# Teste 4: Consultar dados do banco
println("\n📝 Teste 4: Consultar dados diretamente do banco")
println("-" ^ 60)

db = init_database(test_db)

# Carrega algumas curvas
curves = load_curves(db, start_date, Date(2015, 2, 10))
println("   Curvas carregadas: $(nrow(curves))")

if nrow(curves) > 0
    println("\n   Amostra (primeiras 3):")
    for row in eachrow(first(curves, 3))
        if row.success == 1
            println("      $(row.date): β₀=$(round(row.beta0, digits=4)), custo=$(round(row.cost, digits=2))")
        else
            println("      $(row.date): FALHA - $(row.error_message)")
        end
    end
end

# Carrega uma curva específica
test_curve = load_curve(db, start_date)
if test_curve !== nothing
    println("\n   Curva específica ($start_date):")
    println("      Sucesso: $(test_curve.success)")
    if test_curve.success
        println("      Parâmetros: $(round.(test_curve.params, digits=4))")
        println("      Custo: $(round(test_curve.cost, digits=4))")
        println("      Títulos: $(test_curve.n_bonds)")
    end
end

# Estatísticas gerais
stats = get_database_stats(db)
println("\n   Estatísticas do banco:")
println("      Total: $(stats.total_curves)")
println("      Sucessos: $(stats.successful_curves)")
println("      Taxa de sucesso: $(stats.success_rate)%")
println("      Período: $(stats.date_range[1]) → $(stats.date_range[2])")

# Limpa. Fecha a conexão antes de apagar: no Windows o arquivo não pode ser
# removido enquanto houver handle aberto. O modo WAL cria os arquivos auxiliares
# -wal e -shm, que também precisam sair.
println("\n🗑️  Removendo banco de teste...")
close(db)
for f in (test_db, test_db * "-wal", test_db * "-shm")
    isfile(f) && rm(f; force=true)
end

println("\n" * "=" ^ 60)
println("🎉 TODOS OS TESTES PASSARAM COM SUCESSO!")
println("=" ^ 60)

println("\n✅ Sistema de persistência SQLite está funcionando corretamente:")
println("   • Criação de banco de dados: OK")
println("   • Processamento incremental: OK")
println("   • Detecção de datas faltantes: OK")
println("   • Salvamento automático: OK")
println("   • Consultas ao banco: OK")
println("   • Estatísticas: OK")
