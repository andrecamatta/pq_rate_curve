#!/bin/bash

# Script para executar o pipeline completo de teste e geração de resultados.
# 1. Roda a validação cruzada para gerar a configuração ótima.
# 2. Roda o ajuste de curvas para um período específico (2024).
# 3. Gera a animação das curvas de juros.

set -e  # Encerra o script se qualquer comando falhar

echo "🚀 INICIANDO PIPELINE DE TESTE COMPLETO 🚀"
echo "=========================================="

# Etapa 1: Rodar a Validação Cruzada
echo "🔍 Etapa 1/3: Executando a validação cruzada para encontrar a configuração ótima..."
echo "   (Isso pode levar vários minutos...)"
julia run_continuous_walkforward_cv.jl
echo "✅ Validação cruzada concluída. Arquivo 'optimal_config.toml' gerado."
echo "------------------------------------------"

# Etapa 2: Gerar as curvas de juros para 2024
echo "📊 Etapa 2/3: Ajustando as curvas de juros para o ano de 2024..."
julia fit_curvas.jl --start 2024-01-01 --end 2024-12-31
echo "✅ Ajuste de curvas concluído."
echo "------------------------------------------"

# Etapa 3: Gerar a animação
echo "🎬 Etapa 3/3: Gerando a animação das curvas de juros..."

# Encontra o arquivo CSV mais recente gerado pelo script de ajuste
CURVE_FILE=$(ls -t curvas_nss_*.csv 2>/dev/null | head -n 1)

if [ -z "$CURVE_FILE" ]; then
    echo "❌ Erro: Nenhum arquivo de curvas (curvas_nss_*.csv) encontrado."
    exit 1
fi

echo "   Usando o arquivo de curvas: $CURVE_FILE"
julia create_yield_curve_animation.jl "$CURVE_FILE"

# Encontra o arquivo de vídeo mais recente
VIDEO_FILE=$(ls -t yield_curves_animation_*.mp4 2>/dev/null | head -n 1)
echo "✅ Animação concluída. Vídeo salvo como '$VIDEO_FILE'."
echo "------------------------------------------"

echo "🎉 PIPELINE DE TESTE CONCLUÍDO COM SUCESSO! 🎉"
