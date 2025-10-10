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
    fit_nss(
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

## Persistência SQLite (Banco de Dados Histórico)

O módulo oferece sistema de persistência SQLite para armazenar parâmetros NSS históricos, permitindo:
- **Processamento incremental**: Processa apenas datas faltantes
- **Retomada automática**: Interrupções são retomadas automaticamente
- **Consultas eficientes**: SQLite permite análises SQL diretas
- **Backup compacto**: Banco de dados único (~1 MB/ano)

### Modo Banco de Dados com `fit_curves_for_period`

```julia
using PQRateCurve, Dates

# Primeira execução: processa todas as datas e salva no banco
results, config = fit_curves_for_period(
    Date(2015, 1, 1),
    Date(2025, 12, 31);
    db_path="curves.db",       # Ativa persistência SQLite
    output_csv=nothing,         # Opcional: CSV não necessário
    find_continuity=true,
    verbose=true
)

# Segunda execução: apenas processa datas novas!
# (Datas já existentes são carregadas do banco)
results, config = fit_curves_for_period(
    Date(2015, 1, 1),
    Date(2026, 3, 31);  # Estendeu período
    db_path="curves.db",
    verbose=true
)
```

### Funções de Persistência

#### Gerenciamento do Banco

```julia
# Inicializar/abrir banco
db = init_database("curves.db")

# Estatísticas gerais
stats = get_database_stats(db)
println("Total de curvas: $(stats.total_curves)")
println("Taxa de sucesso: $(stats.success_rate)%")
println("Período: $(stats.date_range[1]) → $(stats.date_range[2])")
```

#### Salvar Curvas

```julia
# Salvar curva bem-sucedida
save_curve(db, Date(2024, 1, 2), params, cost, n_bonds, outliers_removed;
          success=true, used_previous_params=false)

# Salvar falha
save_curve_failure(db, Date(2024, 1, 3), "Dados insuficientes")
```

#### Consultar Dados

```julia
# Carregar intervalo de curvas
curves = load_curves(db, Date(2024, 1, 1), Date(2024, 12, 31))

# Carregar curva específica
curve = load_curve(db, Date(2024, 6, 15))
if curve !== nothing && curve.success
    println("β₀ = $(curve.params[1])")
    println("Custo = $(curve.cost)")
end

# Verificar se curva existe
if curve_exists(db, Date(2024, 1, 2))
    println("Curva já processada!")
end

# Obter datas faltantes
missing = get_missing_dates(db, Date(2024, 1, 1), Date(2024, 12, 31))
println("Faltam processar: $(length(missing)) dias")
```

#### Consultar Taxas Diretamente (get_rate)

A função `get_rate` usa **multiple dispatch** do Julia para consultar taxas diretamente do banco, sem reprocessar. Quatro métodos disponíveis:

```julia
db = init_database("historical_curves.db")

# Método 1: Uma data, um prazo → Float64 ou nothing
rate = get_rate(db, Date(2024, 6, 15), 1.0)  # DI 1 ano

if rate !== nothing
    println("DI 1Y: $(round(rate * 100, digits=2))%")
end

# Método 2: Uma data, múltiplos prazos → Vector
rates = get_rate(db, Date(2024, 6, 15), [0.5, 1.0, 2.0, 5.0, 10.0])
println("Estrutura a termo completa:")
for (maturity, r) in zip([0.5, 1.0, 2.0, 5.0, 10.0], rates)
    if r !== nothing
        println("  $(maturity)Y: $(round(r * 100, digits=2))%")
    end
end

# Método 3: Range de datas, um prazo → DataFrame
series = get_rate(db, Date(2024, 1, 1), Date(2024, 12, 31), 1.0)

# Remover datas com dados faltantes
using DataFrames
valid_series = dropmissing(series)

# Plotar evolução temporal
using Plots
plot(valid_series.date, valid_series.rate .* 100,
     xlabel="Data", ylabel="Taxa (%)",
     title="DI 1 ano - Evolução 2024",
     legend=false, lw=2)

# Método 4: Vetor de datas específicas, um prazo → DataFrame
dates = [Date(2024, 1, 2), Date(2024, 4, 1), Date(2024, 7, 1), Date(2024, 10, 1)]
quarterly = get_rate(db, dates, 5.0)  # DI 5 anos trimestral

println(quarterly)
```

**Vantagens do `get_rate`:**
- ✅ **Acesso instantâneo**: Consulta direta ao cache, sem reprocessar
- ✅ **Flexível**: Um ou múltiplos prazos/datas com a mesma função
- ✅ **Type-safe**: Retorna `nothing`/`missing` quando dado não disponível
- ✅ **DataFrame-ready**: Resultados prontos para análise e plotagem

### Exemplo: Construir Base Histórica Completa

```julia
using PQRateCurve, Dates

# Processar todo o período viável (2015-2025)
# Sistema automaticamente:
# - Cria banco se não existir
# - Carrega datas já processadas
# - Processa apenas datas faltantes
# - Salva cada resultado automaticamente

results, config = fit_curves_for_period(
    Date(2015, 2, 1),   # Primeira data viável (90% sucesso)
    Date(2025, 9, 30);  # Última data com dados
    db_path="historical_curves.db",
    find_continuity=true,
    verbose=true
)

# Pode interromper a qualquer momento (Ctrl+C)
# Ao rodar novamente, continua de onde parou!

# Analisar dados do banco
db = init_database("historical_curves.db")
stats = get_database_stats(db)
println("Base histórica: $(stats.total_curves) curvas ($(stats.success_rate)% sucesso)")

# Exportar para CSV se necessário
all_curves = load_curves(db, Date(2015, 1, 1), Date(2025, 12, 31))
CSV.write("curvas_historicas.csv", all_curves)
```

### Schema do Banco de Dados

O banco SQLite contém a tabela `nss_curves`:

| Coluna | Tipo | Descrição |
|--------|------|-----------|
| `date` | TEXT (PK) | Data de referência (yyyy-mm-dd) |
| `beta0, beta1, beta2, beta3` | REAL | Parâmetros β do modelo NSS |
| `tau1, tau2` | REAL | Parâmetros τ do modelo NSS |
| `cost` | REAL | Custo de ajuste (erro médio em R$) |
| `n_bonds` | INTEGER | Número de títulos usados |
| `outliers_removed` | INTEGER | Outliers removidos |
| `success` | INTEGER | 1=sucesso, 0=falha |
| `error_message` | TEXT | Mensagem de erro (se falhou) |
| `used_previous_params` | INTEGER | Usou parâmetros anteriores |
| `reoptimized` | INTEGER | Foi reotimizado |
| `created_at` | TEXT | Timestamp criação |
| `updated_at` | TEXT | Timestamp atualização |

### Consultas SQL Diretas

Você pode usar qualquer ferramenta SQLite para consultas customizadas:

```sql
-- Top 10 piores ajustes
SELECT date, cost, n_bonds FROM nss_curves
WHERE success = 1 ORDER BY cost DESC LIMIT 10;

-- Taxa de sucesso por ano
SELECT strftime('%Y', date) as ano,
       COUNT(*) as total,
       SUM(success) as sucessos,
       ROUND(100.0 * SUM(success) / COUNT(*), 1) as taxa_sucesso
FROM nss_curves
GROUP BY ano ORDER BY ano;

-- Evolução do parâmetro β₀ ao longo do tempo
SELECT date, beta0 FROM nss_curves
WHERE success = 1 ORDER BY date;
```

**Veja exemplo completo em:** [`examples/build_historical_database.jl`](examples/build_historical_database.jl)

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
- `fit_nss(...)` - Otimização completa
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
│   ├── constants.jl             # Constantes do projeto
│   ├── config_service.jl        # Gerenciamento de configuração
│   ├── formatting.jl            # Funções de formatação
│   ├── financial_math.jl        # Funções matemáticas puras
│   ├── data_handling.jl         # Manipulação de dados
│   ├── outlier_detection.jl     # Detecção de outliers
│   ├── estimation.jl            # Otimização e estimação
│   ├── persistence.jl           # Persistência SQLite
│   └── high_level_api.jl        # API de alto nível
├── examples/
│   ├── basic_usage.jl           # Exemplo de uso básico
│   ├── high_level_usage.jl      # Exemplo de alto nível
│   └── build_historical_database.jl  # Construir base histórica SQLite
├── Project.toml                 # Dependências do módulo
└── Manifest.toml                # Lock file de versões
```

## Notas Importantes

1. **Nome do módulo**: O módulo se chama `PQRateCurve` (CamelCase), não `pq_rate_curve`
2. **Dependências**: Todas as dependências estão declaradas em `Project.toml`
3. **Ativação do projeto**: Sempre use `Pkg.activate(".")` no diretório do projeto
4. **Cache**: O módulo usa cache para dados SELIC e dados mensais do BACEN
