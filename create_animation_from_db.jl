#!/usr/bin/env julia
"""
Cria animação de curvas de juros a partir do banco de dados histórico
"""

using PQRateCurve
using CSV, DataFrames, Dates

println("🎬 CRIAÇÃO DE ANIMAÇÃO - BANCO DE DADOS HISTÓRICO")
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

# Salva CSV temporário
temp_csv = "temp_historical_curves.csv"
println("💾 Salvando CSV temporário: $temp_csv...")
CSV.write(temp_csv, df_animation)
println()

# Cria animação
output_video = "historical_curves_animation.mp4"
println("🎥 Criando animação...")
println("   Entrada: $temp_csv")
println("   Saída: $output_video")
println("   Curvas: $(stats.successful_curves) bem-sucedidas")
println("   Período: $(stats.date_range[1]) → $(stats.date_range[2])")
println()

# Chama função de animação com configurações otimizadas
result_path = create_yield_curve_animation(
    temp_csv,
    output_video;
    fps=10,           # 10 frames por segundo
    duration=60,      # 60 segundos de vídeo (para ~2700 curvas)
    config_file="config.toml"
)

# Remove CSV temporário
println()
println("🧹 Removendo arquivo temporário...")
rm(temp_csv)

println()
println("=" ^ 60)
println("✅ ANIMAÇÃO CRIADA COM SUCESSO!")
println("📹 Arquivo: $result_path")
println("=" ^ 60)
