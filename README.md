# PQ Rate Curve

**Módulo Julia** para ajuste de curvas de juros Nelson-Siegel-Svensson (NSS) em títulos públicos brasileiros usando otimização PSO (Particle Swarm Optimization) com refinamento L-BFGS.

> 💡 **Novidade:** Agora disponível como módulo Julia completo! Use `using PQRateCurve` para acessar todas as funções via API ou execute os scripts CLI para processamento em lote.

## Objetivos

Este projeto implementa um sistema robusto para:

1. **Estimação de curvas de juros** usando o modelo Nelson-Siegel-Svensson
2. **Remoção automática de outliers** baseada em threshold fixo de erro e filtro de ultra-baixa liquidez
3. **Otimização híbrida** PSO + L-BFGS para ajuste de parâmetros
4. **Validação cruzada walk-forward** com continuidade temporal
5. **Análise de performance** através de múltiplos regimes econômicos

O sistema processa dados do BACEN (Banco Central do Brasil) e do Tesouro Direto para gerar curvas de juros precisas e estáveis.

## Estrutura do Projeto

```
├── src/
│   ├── PQRateCurve.jl           # Módulo principal
│   ├── config_service.jl        # Gerenciamento centralizado de configurações
│   ├── constants.jl             # Constantes do projeto
│   ├── formatting.jl            # Funções de formatação (números, percentuais, etc.)
│   ├── financial_math.jl        # Funções matemáticas financeiras
│   ├── data_handling.jl         # Manipulação de dados BACEN
│   ├── outlier_detection.jl     # Detecção de outliers (threshold fixo + liquidez)
│   ├── estimation.jl            # Otimização PSO + L-BFGS
│   └── high_level_api.jl        # API de alto nível (fit_curves_for_period, etc.)
├── tests/
│   ├── test_refactoring.jl      # Testes de refatoração
│   ├── test_multiple_dates.jl   # Testes com múltiplas datas
│   └── test_high_level_api.jl   # Testes da API de alto nível
├── examples/
│   └── basic_usage.jl           # Exemplo de uso do módulo
├── outputs/                     # Saídas (CSVs, vídeos) - gitignored
├── config.toml                  # Configuração padrão
├── optimal_config.toml          # Configuração otimizada (gerada)
├── fit_curvas.jl                # Script para ajuste de curvas
├── run_continuous_walkforward_cv.jl # Validação de hiperparâmetros
├── create_yield_curve_animation.jl  # Animação das curvas
├── USO_DO_MODULO.md             # Guia completo de uso como módulo
└── raw/                         # Dados BACEN (zips)
```

## Instalação e Setup

```julia
# Clone o repositório
cd ~/Projetos
git clone <url-do-repo>
cd pq_rate_curve

# Ative o projeto e instale dependências
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

## Uso como Módulo Julia

Este projeto é um **módulo Julia completo** que pode ser usado de duas formas:

### 1. Uso Programático (API)

#### Funções de Alto Nível (Recomendadas)

As funções de alto nível encapsulam workflows completos e são a forma **mais fácil** de usar o módulo:

```julia
using PQRateCurve
using Dates

# Ajustar curvas NSS para um período
results, config = fit_curves_for_period(
    Date(2024, 1, 1),
    Date(2024, 3, 31);
    output_csv="curvas_q1",    # Salva automaticamente
    find_continuity=true,       # Busca parâmetros anteriores
    verbose=true                # Mostra progresso
)

# Analisar resultados
for r in results
    if r.success
        println("$(r.date): β₀=$(round(r.params[1], digits=4)), custo=$(round(r.cost, digits=6))")
    end
end

# Criar animação das curvas
video = create_yield_curve_animation(
    "curvas_q1_2024-01-01_12-00-00.csv",
    "animacao_q1.mp4";
    fps=15,
    duration=20
)
```

**📖 Ver documentação completa de alto nível:** [`USO_DO_MODULO.md`](USO_DO_MODULO.md#funções-de-alto-nível-recomendadas)

**🎓 Ver exemplo completo:** [`examples/high_level_usage.jl`](examples/high_level_usage.jl)

#### Funções de Baixo Nível

Para controle fino ou uso customizado:

```julia
using PQRateCurve
using Dates

# Calcular taxas NSS para diferentes prazos
params = [0.10, -0.02, -0.01, 0.005, 5.0, 15.0]  # β0, β1, β2, β3, τ1, τ2
taxa_1y = nss_rate(1.0, params)   # Taxa para 1 ano
taxa_5y = nss_rate(5.0, params)   # Taxa para 5 anos

# Precificar um título LTN
ref_date = Date(2024, 1, 15)
maturity_date = Date(2025, 1, 15)
cash_flow = [(maturity_date, 1000.0)]
preco = price_bond(cash_flow, ref_date, params)

# Calcular duration
duration = calculate_duration(cash_flow, ref_date, params)

# Carregar dados do BACEN e otimizar
df = load_bacen_data(ref_date, ref_date)
cash_flows, quantities, info = generate_cash_flows_with_quantity(df, ref_date)

# Otimização completa com remoção de outliers
config = load_configuration("config.toml")
params_otimos, custo, flows_limpos, outliers, iters =
    fit_nss(
        cash_flows, ref_date,
        config["pso"]["lower_bounds"],
        config["pso"]["upper_bounds"];
        bond_quantities=quantities,
        pso_N=config["pso"]["N"]
    )
```

**🎓 Ver exemplo de baixo nível:** [`examples/basic_usage.jl`](examples/basic_usage.jl)

#### Principais Funções Exportadas

**Alto Nível:**
- `fit_curves_for_period(start, end; options...)` - Ajusta curvas para período
- `create_yield_curve_animation(csv, video; options...)` - Cria animação

> **Nota:** Validação de hiperparâmetros é feita via script CLI `run_continuous_walkforward_cv.jl` (processamento paralelo distribuído - ver seção Scripts CLI abaixo)

**Matemática Financeira:**
- `nss_rate(t, params)` - Taxa NSS para prazo t
- `price_bond(cash_flow, ref_date, params)` - Precificação de títulos
- `calculate_duration(...)` - Cálculo de duration
- `calculate_ytm(...)` - Yield to maturity

**Manipulação de Dados:**
- `load_bacen_data(start_date, end_date)` - Carregar dados BACEN
- `generate_cash_flows_with_quantity(...)` - Gerar cash flows
- `load_configuration(file)` - Carregar config TOML (legacy)

**Gerenciamento de Configurações (ConfigService):**
- `load_config(file)` - Carregar configuração com validação
- `get_cv_config(service)` - Obter configurações de cross-validation
- `get_pso_config(service)` - Obter configurações PSO
- `get_pso_bounds(config)` - Obter bounds PSO (single source of truth)

**Funções de Formatação:**
- `format_percentage(x)` - Formatar como porcentagem
- `format_score(x)` - Formatar scores/custos
- `format_nss_params(params)` - Formatar parâmetros NSS
- `format_currency(x)` - Formatar valores monetários
- Mais 10 funções de formatação especializadas

**Otimização:**
- `fit_nss(...)` - Otimização completa
- `detect_outliers_mad_and_liquidity(...)` - Detecção de outliers
- `refine_nss_with_lbfgs(...)` - Refinamento L-BFGS
- `normalize_cost_by_volume(dates, costs)` - Normalização de custos por volume

### 2. Scripts de Linha de Comando

Os scripts principais (`fit_curvas.jl`, `run_continuous_walkforward_cv.jl`, etc.) usam o módulo internamente e podem ser executados diretamente:

## Como Usar (Scripts CLI)

### 1. Validação de Hiperparâmetros

Para estabelecer os hiperparâmetros ótimos através de validação cruzada walk-forward:

```bash
julia --project=. run_continuous_walkforward_cv.jl
```

**O que faz:**
- Executa otimização bayesiana sobre 6 regimes econômicos diferentes (2015-2024)
- Testa configurações PSO vs PSO+L-BFGS através de 20 configurações
- Avalia performance usando blocos de 30 dias (treino → teste)
- Gera arquivo `optimal_config.toml` com a melhor configuração

**Configuração (config.toml):**
```toml
[validation]
num_hyperparameter_configs = 20  # Número de configurações testadas

[pso]
N = 80                    # Número de partículas
C1 = 2.92                 # Aceleração cognitiva
C2 = 1.68                 # Aceleração social
omega = 0.57              # Peso de inércia
f_calls_limit = 1500      # Limite de avaliações

[optimization]
use_lbfgs = true          # Usar refinamento L-BFGS
temporal_penalty_weight = 0.1978  # Penalidade de continuidade temporal

[outlier_detection]
error_threshold_global = 27.67    # Threshold fixo de erro (R$)
ultra_low_factor = 4.96           # Fator de ultra-baixa liquidez
fator_liq = 0.0111                # Fator de liquidez (1.11% do volume)
```

### 2. Ajuste de Curvas para Intervalo de Datas

Para ajustar curvas NSS em um período específico:

```bash
julia --project=. fit_curvas.jl --start 2024-01-01 --end 2024-12-31
```

**Opções disponíveis:**
```bash
julia --project=. fit_curvas.jl --start YYYY-MM-DD --end YYYY-MM-DD [opções]

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
julia --project=. fit_curvas.jl --start 2024-01-01 --end 2024-12-31

# Apenas primeiro trimestre de 2024
julia --project=. fit_curvas.jl --start 2024-01-01 --end 2024-03-31

# Teste de configuração sem executar
julia --project=. fit_curvas.jl --start 2024-01-01 --end 2024-01-31 --dry-run
```

**O que faz:**
- Carrega dados BACEN para cada data útil no período
- Aplica detecção de outliers (threshold fixo + filtro de ultra-baixa liquidez)
- Otimiza parâmetros NSS usando PSO + L-BFGS (se configurado)
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
julia --project=. create_yield_curve_animation.jl curvas_nss_2024-01-01_12-00-00.csv [output_video.mp4]
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
2. **L-BFGS**: Refinamento local para convergência precisa
3. **Continuidade temporal**: Penalidade para mudanças bruscas entre dias consecutivos

### Detecção de Outliers

- **Threshold Fixo de Erro**: Remove títulos com erro de precificação acima de threshold configurável (ex: 20 R$)
- **Filtro de Ultra-Baixa Liquidez**: Remove títulos com volume extremamente baixo (< factor × mediana)
- **Critérios independentes**: Títulos são removidos se satisfizerem qualquer um dos critérios
- **Configuração flexível**: Parâmetros `error_threshold_global` e `ultra_low_factor` ajustáveis via config.toml

### Validação Cruzada

- **Walk-forward**: Treina em período passado, testa em período futuro
- **Regimes econômicos**: Crise 2015, Recessão 2016, Pandemia 2020, etc.
- **Métricas**: Custo out-of-sample, overfitting ratio, estabilidade dos vértices

## Requisitos

### Julia (versão 1.11+)

Este é um **módulo Julia completo**. Todas as dependências estão declaradas em `Project.toml`:
```julia
# Principais dependências
CSV, DataFrames, Dates, HTTP, ZipFile
Optim, Metaheuristics, Hyperopt
Plots, Statistics, LinearAlgebra
TOML, JSON, BusinessDays
```

**Instalação das dependências:**
```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()  # Instala todas as dependências do Project.toml
```

### Dados

O sistema baixa automaticamente dados do BACEN quando necessário, mas você pode pré-carregar arquivos ZIP na pasta `raw/`.

## Exemplo de Fluxo Completo

### Usando Scripts CLI

```bash
# 1. Validar hiperparâmetros (uma vez)
julia --project=. run_continuous_walkforward_cv.jl

# 2. Ajustar curvas para 2024
julia --project=. fit_curvas.jl --start 2024-01-01 --end 2024-12-31

# 3. Criar animação
julia --project=. create_yield_curve_animation.jl curvas_nss_2024-01-01_12-00-00.csv animacao_2024.mp4
```

### Usando o Módulo Programaticamente

```julia
using PQRateCurve

# Exemplo rápido: calcular taxa para 1 ano
params = [0.10, -0.02, -0.01, 0.005, 5.0, 15.0]
taxa = nss_rate(1.0, params)
println("Taxa 1 ano: $(round(taxa*100, digits=2))%")

# Ver examples/basic_usage.jl para exemplo completo
```

## Performance

**Benchmarks típicos:**
- Validação completa (20 configs): ~15-25 minutos
- Ajuste anual (250 dias): ~8-12 minutos
- Animação: ~30-60 segundos

## Testes

O projeto inclui uma suíte completa de testes localizada em `tests/`:

```bash
# Executar teste básico de refatoração
julia --project=. tests/test_refactoring.jl

# Executar teste com múltiplas datas
julia --project=. tests/test_multiple_dates.jl

# Executar teste da API de alto nível
julia --project=. tests/test_high_level_api.jl
```

**Cobertura dos testes:**
- ✅ Funções de formatação (14 funções)
- ✅ Fit de curva NSS (PSO + L-BFGS)
- ✅ Detecção de outliers
- ✅ Continuidade temporal
- ✅ Normalização de custos
- ✅ API de alto nível
- ✅ Integração completa

## Referências

- Nelson, C.R. & Siegel, A.F. (1987). "Parsimonious Modeling of Yield Curves"
- Svensson, L.E.O. (1994). "Estimating and Interpreting Forward Interest Rates"
- Kennedy, J. & Eberhart, R. (1995). "Particle Swarm Optimization"