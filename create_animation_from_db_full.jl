#!/usr/bin/env julia
"""
Cria animação COMPLETA de curvas de juros - 1 FRAME POR CURVA
"""

using PQRateCurve
using CSV, DataFrames, Dates

println("🎬 CRIAÇÃO DE ANIMAÇÃO COMPLETA - 1 FRAME POR CURVA")
println("=" ^ 60)

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
println("📥 Carregando todas as curvas do banco...")
df_curves = load_curves(db, stats.date_range[1], stats.date_range[2])
println("✅ Carregadas $(nrow(df_curves)) curvas")
println()

# Converte para formato CSV esperado pela função de animação
println("🔄 Convertendo para formato de animação...")
df_animation = DataFrame(
    Data = df_curves.date,
    Sucesso = df_curves.success .== 1,
    Beta0 = df_curves.beta0,
    Beta1 = df_curves.beta1,
    Beta2 = df_curves.beta2,
    Beta3 = df_curves.beta3,
    Tau1 = df_curves.tau1,
    Tau2 = df_curves.tau2,
    Custo = df_curves.cost,
    NumTitulos = df_curves.n_bonds,
    OutliersRemovidos = df_curves.outliers_removed,
    UsouPreviousParams = df_curves.used_previous_params .== 1,
    ErroMensagem = df_curves.error_message
)

# Calcula duração para 1 frame por curva
fps = 15  # 15 FPS = velocidade confortável
successful_curves = stats.successful_curves
duration_seconds = ceil(Int, successful_curves / fps)
duration_minutes = round(duration_seconds / 60, digits=1)

println("⏱️  CONFIGURAÇÃO DO VÍDEO:")
println("   FPS: $fps")
println("   Frames: $successful_curves (1 por curva)")
println("   Duração: $duration_seconds segundos (~$duration_minutes minutos)")
println()

# Salva CSV temporário
temp_csv = "temp_historical_curves_full.csv"
println("💾 Salvando CSV temporário: $temp_csv...")
CSV.write(temp_csv, df_animation)
println()

# Cria animação COMPLETA
output_video = "historical_curves_animation_FULL.mp4"
println("🎥 Criando animação completa...")
println("   Entrada: $temp_csv")
println("   Saída: $output_video")
println("   Modo: 1 FRAME POR CURVA (sem pulos)")
println()

# Chama função de animação com duração calculada
# Nota: NÃO passar config_file para usar nossos parâmetros customizados
result_path = create_yield_curve_animation(
    temp_csv,
    output_video;
    fps=fps,
    duration=duration_seconds  # Duração exata para 1 frame por curva
)

# Remove CSV temporário
println()
println("🧹 Removendo arquivo temporário...")
rm(temp_csv)

println()
println("=" ^ 60)
println("✅ ANIMAÇÃO COMPLETA CRIADA COM SUCESSO!")
println("📹 Arquivo: $result_path")
println("⏱️  Duração: ~$duration_minutes minutos")
println("🎬 $(successful_curves) frames (1 por curva)")
println("=" ^ 60)
