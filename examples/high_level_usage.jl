#!/usr/bin/env julia --project=..

"""
high_level_usage.jl - Exemplo de uso das funções de alto nível do PQRateCurve

Este exemplo demonstra como usar as funções de alto nível para:
1. Ajustar curvas NSS para um período
2. Analisar resultados
3. Criar animação das curvas

Execute: julia --project=.. examples/high_level_usage.jl
"""

using PQRateCurve
using Dates

println("=" ^ 70)
println("Exemplo de Uso de Alto Nível - PQRateCurve")
println("=" ^ 70)

# ============================================================================
# 1. Ajustar curvas NSS para um período curto (exemplo: 5 dias úteis)
# ============================================================================

println("\\n📊 1. Ajustando curvas NSS para primeiros 5 dias úteis de 2024...")

# Ajusta curvas de 02/01 a 08/01/2024 (5 dias úteis)
results, config = fit_curves_for_period(
    Date(2024, 1, 2),
    Date(2024, 1, 8);
    config_file="../config.toml",
    output_csv="exemplos_curvas",
    find_continuity=false,  # Sem continuidade para exemplo rápido
    verbose=false  # Sem output verbose para deixar exemplo limpo
)

println("\\n✅ Fit concluído!")
println("   Total de datas: $(length(results))")
println("   Sucessos: $(sum(r.success for r in results))")

# ============================================================================
# 2. Analisar resultados individuais
# ============================================================================

println("\\n📈 2. Analisando resultados individuais...")

for (i, result) in enumerate(results)
    if result.success
        β0, β1, β2, β3, τ1, τ2 = result.params
        println("\\n   Dia $i - $(result.date):")
        println("      Parâmetros NSS:")
        println("         β₀ = $(round(β0, digits=4))  (nível de longo prazo)")
        println("         β₁ = $(round(β1, digits=4))  (inclinação)")
        println("         β₂ = $(round(β2, digits=4))  (curvatura curto prazo)")
        println("         β₃ = $(round(β3, digits=4))  (curvatura longo prazo)")
        println("         τ₁ = $(round(τ1, digits=2))  (decaimento curto)")
        println("         τ₂ = $(round(τ2, digits=2))  (decaimento longo)")
        println("      Custo: $(round(result.cost, digits=6))")
        println("      Títulos usados: $(result.n_bonds)")
        if result.outliers_removed > 0
            println("      Outliers removidos: $(result.outliers_removed)")
        end

        # Calcula taxas para diferentes prazos
        prazos = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
        println("      Taxas estimadas:")
        for prazo in prazos
            taxa = nss_rate(prazo, result.params) * 100
            println("         $(prazo) anos: $(round(taxa, digits=2))%")
        end
    else
        println("\\n   Dia $i - $(result.date): ❌ Falhou")
        if result.error_message !== nothing
            println("      Erro: $(result.error_message)")
        end
    end
end

# ============================================================================
# 3. Calcular métricas agregadas
# ============================================================================

println("\\n📊 3. Métricas agregadas do período...")

successful_results = filter(r -> r.success, results)

if !isempty(successful_results)
    # Média de títulos por dia
    avg_bonds = mean(r.n_bonds for r in successful_results)
    println("   Média de títulos por dia: $(round(avg_bonds, digits=1))")

    # Taxa de remoção de outliers
    total_outliers = sum(r.outliers_removed for r in successful_results)
    println("   Total de outliers removidos: $total_outliers")

    # Custo médio
    avg_cost = mean(r.cost for r in successful_results)
    println("   Custo médio: $(round(avg_cost, digits=6))")

    # Estabilidade dos parâmetros (desvio padrão)
    beta0_values = [r.params[1] for r in successful_results]
    beta0_std = std(beta0_values)
    println("   Estabilidade β₀: σ=$(round(beta0_std, digits=6))")
end

# ============================================================================
# 4. Usar função fit com opções diferentes
# ============================================================================

println("\\n🔧 4. Exemplo de fit com configurações customizadas...")

# Sem salvar CSV, apenas retornar resultados
results2, _ = fit_curves_for_period(
    Date(2024, 1, 2),
    Date(2024, 1, 5);
    output_csv=nothing,  # Não salva CSV
    find_continuity=false,
    verbose=false
)

println("   Processados $(length(results2)) dias sem salvar CSV")

# ============================================================================
# 5. Criar animação (se houver CSV gerado)
# ============================================================================

println("\\n🎬 5. Criando animação das curvas...")

# Procura o CSV que foi gerado
using Glob
csv_files = glob("exemplos_curvas_*.csv")

if !isempty(csv_files)
    latest_csv = last(sort(csv_files))
    println("   Usando CSV: $latest_csv")

    # Cria animação curta (5 segundos, 10 FPS)
    try
        video_path = create_yield_curve_animation(
            latest_csv,
            "exemplo_animacao.mp4";
            fps=10,
            duration=5,  # Vídeo curto para exemplo
            maturities=[0.25, 0.5, 1.0, 2.0, 5.0, 10.0]  # Menos pontos
        )

        println("\\n✅ Animação criada: $video_path")
    catch e
        println("⚠️ Não foi possível criar animação: $e")
    end
else
    println("   ⚠️ Nenhum CSV encontrado, pulando animação")
end

println("\\n" ^ "=" * 70)
println("✅ Exemplo concluído!")
println("\\nPróximos passos:")
println("  - Ajuste curvas para períodos maiores")
println("  - Experimente com diferentes configurações")
println("  - Analise a estabilidade temporal dos parâmetros")
println("  - Use os resultados para precificação de títulos")
println("=" ^ 70)
