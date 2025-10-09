# Usando PQRateCurve como Módulo Julia

Este repositório está configurado como um módulo Julia completo e pode ser usado de duas formas:

## 1. Uso Local (Desenvolvimento)

Para usar o módulo localmente durante o desenvolvimento:

```julia
# Ative o projeto
cd("/caminho/para/pq_rate_curve")
using Pkg
Pkg.activate(".")

# Importe o módulo
using PQRateCurve

# Use as funções exportadas
params = [0.10, -0.02, -0.01, 0.005, 5.0, 15.0]
rate = nss_rate(1.0, params)  # Taxa para 1 ano
```

## 2. Uso como Dependência

Para usar este módulo em outro projeto Julia:

### Opção A: Instalação via caminho local

```julia
using Pkg
Pkg.develop(path="/caminho/para/pq_rate_curve")
```

### Opção B: Instalação via Git (se publicado)

```julia
using Pkg
Pkg.add(url="https://github.com/seu-usuario/pq_rate_curve")
```

## Exemplo Básico

```julia
using PQRateCurve
using Dates

# 1. Calcular taxas NSS
params = [0.10, -0.02, -0.01, 0.005, 5.0, 15.0]
rate_1y = nss_rate(1.0, params)  # Taxa para 1 ano
rate_5y = nss_rate(5.0, params)  # Taxa para 5 anos

# 2. Precificar um título LTN
ref_date = Date(2024, 1, 15)
maturity_date = Date(2025, 1, 15)
cash_flow = [(maturity_date, 1000.0)]  # Valor de face
price = price_bond(cash_flow, ref_date, params)

# 3. Calcular duration
duration = calculate_duration(cash_flow, ref_date, params)

# 4. Carregar dados do BACEN
df = load_bacen_data(ref_date, ref_date)

# 5. Otimizar parâmetros NSS
cash_flows, quantities, bond_info = generate_cash_flows_with_quantity(df, ref_date)
config = load_configuration("config.toml")
lower_bounds = config["pso"]["lower_bounds"]
upper_bounds = config["pso"]["upper_bounds"]

optimal_params, cost, clean_flows, outliers, iters =
    optimize_nelson_siegel_svensson_with_mad_outlier_removal(
        cash_flows, ref_date, lower_bounds, upper_bounds;
        bond_quantities=quantities,
        pso_N=config["pso"]["N"]
    )
```

## Funções de Alto Nível (Recomendadas)

As funções de alto nível encapsulam workflows completos e são **a forma mais fácil** de usar o módulo:

### `fit_curves_for_period(start_date, end_date; options...)`

Ajusta curvas NSS para todos os dias úteis em um período.

```julia
using PQRateCurve, Dates

# Ajustar curvas para Q1 2024
results, config = fit_curves_for_period(
    Date(2024, 1, 1),
    Date(2024, 3, 31);
    output_csv="curvas_q1",  # Salva automaticamente
    find_continuity=true,     # Busca parâmetros anteriores
    verbose=true              # Mostra progresso
)

# Analisar resultados
for r in results
    if r.success
        println("$(r.date): β₀=$(round(r.params[1], digits=4)), custo=$(round(r.cost, digits=6))")
    end
end
```

**Opções:**
- `config_file::String="config.toml"` - Arquivo de configuração
- `output_csv::Union{String,Nothing}="curvas_nss"` - Nome base do CSV (ou `nothing` para não salvar)
- `find_continuity::Bool=true` - Buscar parâmetros anteriores para continuidade temporal
- `verbose::Bool=true` - Mostrar progresso

**Retorna:**
- `results::Vector{DayResult}` - Vetor com resultados de cada dia
- `config::Dict` - Configuração usada

**Struct DayResult:**
```julia
struct DayResult
    date::Date
    success::Bool
    params::Union{Vector{Float64}, Nothing}  # [β0, β1, β2, β3, τ1, τ2]
    cost::Union{Float64, Nothing}
    n_bonds::Int
    outliers_removed::Int
    error_message::Union{String, Nothing}
    used_previous_params::Bool
    reoptimized::Bool
end
```

### `create_yield_curve_animation(csv_file, output_video; options...)`

Cria vídeo animado das curvas de juros a partir do CSV gerado por `fit_curves_for_period`.

```julia
# Criar animação de 20 segundos a 15 FPS
video = create_yield_curve_animation(
    "curvas_q1_2024-01-01_12-00-00.csv",
    "animacao_q1.mp4";
    fps=15,
    duration=20,
    maturities=[0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
)

println("Vídeo criado: $video")
```

**Opções:**
- `fps::Int=10` - Frames por segundo
- `duration::Int=30` - Duração do vídeo em segundos
- `maturities::Vector{Float64}=[0.25, 0.5, ..., 10.0]` - Prazos a plotar
- `plot_size::Tuple{Int,Int}=(1200, 800)` - Tamanho do plot
- `plot_dpi::Int=200` - Resolução DPI
- `config_file::Union{String,Nothing}=nothing` - Config TOML opcional

**Retorna:**
- `String` - Caminho do vídeo gerado

### Exemplo Completo de Workflow

```julia
using PQRateCurve, Dates

# 1. Ajustar curvas para período
results, config = fit_curves_for_period(
    Date(2024, 1, 1),
    Date(2024, 12, 31);
    output_csv="curvas_2024"
)

# 2. Análise dos resultados
successful = filter(r -> r.success, results)
println("Taxa de sucesso: $(length(successful))/$(length(results))")

# Estatísticas dos parâmetros
β0_values = [r.params[1] for r in successful]
println("β₀ médio: $(mean(β0_values))")
println("β₀ σ: $(std(β0_values))")

# 3. Criar animação
csv_files = glob("curvas_2024_*.csv")
if !isempty(csv_files)
    video = create_yield_curve_animation(
        last(csv_files),
        "animacao_2024.mp4"
    )
end
```

**Veja exemplo completo em:** [`examples/high_level_usage.jl`](examples/high_level_usage.jl)

## Funções de Baixo Nível

Para controle mais fino ou uso customizado, você pode usar as funções de baixo nível:

### Matemática Financeira
- `nss_rate(t, params)` - Calcula taxa NSS para prazo t
- `price_bond(cash_flow, ref_date, params)` - Precifica título
- `calculate_duration(cash_flow, ref_date, params)` - Calcula duration
- `calculate_ytm(market_price, cash_flow, ref_date)` - Calcula YTM
- `yearfrac(start_date, end_date)` - Calcula fração de ano (ACT/252)

### Manipulação de Dados
- `load_bacen_data(start_date, end_date)` - Carrega dados do BACEN
- `generate_cash_flows_with_quantity(df, ref_date)` - Gera cash flows
- `load_configuration(config_file)` - Carrega configuração
- `save_optimal_configuration(config, performance, output_file)` - Salva config ótima

### Detecção de Outliers
- `detect_outliers_mad_and_liquidity(...)` - Detecta outliers
- `calculate_mad(values)` - Calcula MAD

### Otimização
- `optimize_nelson_siegel_svensson_with_mad_outlier_removal(...)` - Otimização completa
- `refine_nss_with_lbfgs(...)` - Refinamento L-BFGS
- `calculate_out_of_sample_cost_reais(...)` - Calcula custo out-of-sample
- `precompute_cash_flow_times(cash_flows, ref_date)` - Pré-calcula tempos

### Validação Cross-regime
- `run_walkforward_validation(...)` - Validação walk-forward
- `generate_pso_configs(...)` - Gera configurações PSO
- `generate_focused_pso_configs(...)` - Gera configs focadas

## Scripts de Exemplo

Veja `examples/basic_usage.jl` para um exemplo completo de uso do módulo.

## Scripts Principais

O repositório inclui scripts prontos para uso:

- `fit_curvas.jl` - Ajusta curvas NSS para um período
- `run_continuous_walkforward_cv.jl` - Validação cruzada walk-forward
- `create_yield_curve_animation.jl` - Cria animação das curvas

Todos os scripts agora usam o módulo via `using PQRateCurve`.

## Estrutura do Módulo

```
pq_rate_curve/
├── src/
│   ├── PQRateCurve.jl          # Módulo principal
│   ├── financial_math.jl        # Funções matemáticas puras
│   ├── data_handling.jl         # Manipulação de dados
│   ├── outlier_detection.jl     # Detecção de outliers
│   └── estimation.jl            # Otimização e estimação
├── examples/
│   └── basic_usage.jl           # Exemplo de uso
├── Project.toml                 # Dependências do módulo
└── Manifest.toml                # Lock file de versões
```

## Notas Importantes

1. **Nome do módulo**: O módulo se chama `PQRateCurve` (CamelCase), não `pq_rate_curve`
2. **Dependências**: Todas as dependências estão declaradas em `Project.toml`
3. **Ativação do projeto**: Sempre use `Pkg.activate(".")` no diretório do projeto
4. **Cache**: O módulo usa cache para dados SELIC e dados mensais do BACEN
