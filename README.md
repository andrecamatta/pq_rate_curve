# PQ Rate Curve

Sistema de ajuste de curvas de juros Nelson-Siegel-Svensson (NSS) para títulos públicos brasileiros usando otimização PSO (Particle Swarm Optimization) com refinamento Levenberg-Marquardt.

## Objetivos

Este projeto implementa um sistema robusto para:

1. **Estimação de curvas de juros** usando o modelo Nelson-Siegel-Svensson
2. **Remoção automática de outliers** baseada em MAD (Median Absolute Deviation) e critérios de liquidez
3. **Otimização híbrida** PSO + Levenberg-Marquardt para ajuste de parâmetros
4. **Validação cruzada walk-forward** com continuidade temporal
5. **Análise de performance** através de múltiplos regimes econômicos

O sistema processa dados do BACEN (Banco Central do Brasil) e do Tesouro Direto para gerar curvas de juros precisas e estáveis.

## Estrutura do Projeto

```
├── src/
│   ├── PQRateCurve.jl           # Módulo principal
│   ├── financial_math.jl        # Funções matemáticas financeiras
│   ├── data_handling.jl         # Manipulação de dados BACEN
│   ├── outlier_detection.jl     # Detecção de outliers MAD + liquidez
│   └── estimation.jl            # Otimização PSO + LM
├── config.toml                  # Configuração padrão
├── optimal_config.toml          # Configuração otimizada (gerada)
├── fit_curvas.jl               # Script para ajuste de curvas
├── run_continuous_walkforward_cv.jl # Validação de hiperparâmetros
├── create_yield_curve_animation.jl # Animação das curvas
└── raw/                        # Dados BACEN (zips)
```

## Como Usar

### 1. Validação de Hiperparâmetros

Para estabelecer os hiperparâmetros ótimos através de validação cruzada walk-forward:

```bash
julia run_continuous_walkforward_cv.jl
```

**O que faz:**
- Executa otimização bayesiana sobre 6 regimes econômicos diferentes (2015-2024)
- Testa configurações PSO vs PSO+LM através de 20 configurações
- Avalia performance usando blocos de 30 dias (treino → teste)
- Gera arquivo `optimal_config.toml` com a melhor configuração

**Configuração (config.toml):**
```toml
[validation]
num_hyperparameter_configs = 20  # Número de configurações testadas

[pso]
N = 80                    # Número de partículas
C1 = 1.8                  # Aceleração cognitiva
C2 = 1.3                  # Aceleração social
omega = 0.45              # Peso de inércia
f_calls_limit = 1500      # Limite de avaliações

[optimization]
use_lm = true             # Usar refinamento Levenberg-Marquardt
temporal_penalty_weight = 0.01  # Penalidade de continuidade temporal

[outlier_detection]
mad_threshold = 12.0      # Threshold MAD (sigma)
fator_liq = 0.015         # Fator de liquidez (1.5% do volume)
```

### 2. Ajuste de Curvas para Intervalo de Datas

Para ajustar curvas NSS em um período específico:

```bash
julia fit_curvas.jl --start 2024-01-01 --end 2024-12-31
```

**Opções disponíveis:**
```bash
julia fit_curvas.jl --start YYYY-MM-DD --end YYYY-MM-DD [opções]

--start          Data inicial (YYYY-MM-DD) [padrão: 2024-01-01]
--end            Data final (YYYY-MM-DD) [padrão: 2024-12-31]  
--output         Prefixo do arquivo de saída [padrão: curvas_nss]
--continuity     Usar continuidade temporal [padrão: true]
--verbose        Mostrar progresso detalhado [padrão: true]
--dry-run        Apenas validar configuração, não executar
```

**Exemplos:**
```bash
# Ajustar todo o ano de 2024
julia fit_curvas.jl --start 2024-01-01 --end 2024-12-31

# Apenas primeiro trimestre de 2024
julia fit_curvas.jl --start 2024-01-01 --end 2024-03-31

# Teste de configuração sem executar
julia fit_curvas.jl --start 2024-01-01 --end 2024-01-31 --dry-run
```

**O que faz:**
- Carrega dados BACEN para cada data útil no período
- Aplica detecção de outliers (MAD + liquidez)
- Otimiza parâmetros NSS usando PSO + LM (se configurado)
- Mantém continuidade temporal usando parâmetros do dia anterior
- Gera arquivo CSV com resultados: `curvas_nss_YYYY-MM-DD_HH-MM-SS.csv`

**Arquivo de saída:**
```csv
Data,Sucesso,Beta0,Beta1,Beta2,Beta3,Tau1,Tau2,Custo,NumTitulos,OutliersRemovidos,UsouPreviousParams,Reotimizado,ErroMensagem
2024-01-02,true,0.1234,0.0567,-0.0123,0.0089,1.5432,3.2156,0.0045,15,2,false,false,
```

### 3. Criação de Animação das Curvas

Para criar uma animação das curvas de juros ao longo do tempo:

```bash
julia create_yield_curve_animation.jl curvas_nss_2024-01-01_12-00-00.csv [output_video.mp4]
```

**Parâmetros:**
- `curvas_nss_*.csv`: Arquivo CSV com os parâmetros NSS
- `output_video.mp4` (opcional): Nome do vídeo (padrão: timestamp automático)

**O que faz:**
- Lê parâmetros NSS do arquivo CSV
- Gera curvas de juros para prazos de 0.25 a 10 anos
- Cria animação mostrando evolução temporal das curvas
- Salva vídeo MP4 com 30 segundos de duração (10 FPS)

**Configuração da animação:**
```julia
const FPS = 10        # frames por segundo
const DURATION = 30   # duração em segundos
```

## Metodologia Técnica

### Modelo Nelson-Siegel-Svensson

Curva de juros parametrizada por 6 parâmetros:
```
r(τ) = β₀ + β₁[(1-e^(-τ/τ₁))/(τ/τ₁)] + β₂[((1-e^(-τ/τ₁))/(τ/τ₁)) - e^(-τ/τ₁)] + β₃[((1-e^(-τ/τ₂))/(τ/τ₂)) - e^(-τ/τ₂)]
```

### Otimização Híbrida

1. **PSO (Particle Swarm Optimization)**: Exploração global do espaço de parâmetros
2. **Levenberg-Marquardt**: Refinamento local para convergência precisa
3. **Continuidade temporal**: Penalidade para mudanças bruscas entre dias consecutivos

### Detecção de Outliers

- **MAD (Median Absolute Deviation)**: Remove títulos com preços anômalos
- **Critério de liquidez**: Remove títulos com baixo volume negociado
- **Critérios simultâneos**: Outlier apenas se AMBAS condições são verdadeiras

### Validação Cruzada

- **Walk-forward**: Treina em período passado, testa em período futuro
- **Regimes econômicos**: Crise 2015, Recessão 2016, Pandemia 2020, etc.
- **Métricas**: Custo out-of-sample, overfitting ratio, estabilidade dos vértices

## Requisitos

### Julia (versão 1.6+)

Pacotes necessários (ver `Project.toml`):
```julia
using CSV, DataFrames, Dates, HTTP, ZipFile
using Optim, Metaheuristics, Hyperopt
using Plots, Statistics, LinearAlgebra
using TOML, JSON, BusinessDays
```

### Dados

O sistema baixa automaticamente dados do BACEN quando necessário, mas você pode pré-carregar arquivos ZIP na pasta `raw/`.

## Exemplo de Fluxo Completo

```bash
# 1. Validar hiperparâmetros (uma vez)
julia run_continuous_walkforward_cv.jl

# 2. Ajustar curvas para 2024
julia fit_curvas.jl --start 2024-01-01 --end 2024-12-31

# 3. Criar animação
julia create_yield_curve_animation.jl curvas_nss_2024-01-01_12-00-00.csv animacao_2024.mp4
```

## Performance

**Benchmarks típicos:**
- Validação completa (20 configs): ~15-25 minutos
- Ajuste anual (250 dias): ~8-12 minutos  
- Animação: ~30-60 segundos

## Referências

- Nelson, C.R. & Siegel, A.F. (1987). "Parsimonious Modeling of Yield Curves"
- Svensson, L.E.O. (1994). "Estimating and Interpreting Forward Interest Rates"
- Kennedy, J. & Eberhart, R. (1995). "Particle Swarm Optimization"