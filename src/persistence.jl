"""
    persistence.jl

Módulo de persistência SQLite para parâmetros de curvas NSS.

Armazena histórico de ajustes de curvas em banco de dados SQLite,
permitindo processamento incremental e consultas eficientes.
"""

using SQLite
using Dates
using DataFrames

# ============================================================================
# Schema e Inicialização
# ============================================================================

"""
    init_database(db_path::String) -> SQLite.DB

Cria ou abre banco de dados SQLite e garante schema correto.

# Schema
- `nss_curves`: Parâmetros NSS e metadados por data
  - `date`: Data de referência (PRIMARY KEY)
  - `beta0, beta1, beta2, beta3, tau1, tau2`: Parâmetros NSS
  - `cost`: Custo de ajuste (erro médio em R\$)
  - `n_bonds`: Número de títulos usados
  - `outliers_removed`: Quantidade de outliers removidos
  - `success`: Booleano indicando sucesso do ajuste
  - `error_message`: Mensagem de erro (se falhou)
  - `used_previous_params`: Se usou parâmetros anteriores como seed
  - `reoptimized`: Se foi reotimizado
  - `created_at`: Timestamp de criação
  - `updated_at`: Timestamp de última atualização

# Exemplo
```julia
db = init_database("curves.db")
```
"""
function init_database(db_path::String)
    db = SQLite.DB(db_path)

    # Cria tabela se não existir
    SQLite.execute(db, """
        CREATE TABLE IF NOT EXISTS nss_curves (
            date TEXT PRIMARY KEY,
            beta0 REAL,
            beta1 REAL,
            beta2 REAL,
            beta3 REAL,
            tau1 REAL,
            tau2 REAL,
            cost REAL,
            n_bonds INTEGER,
            outliers_removed INTEGER,
            success INTEGER,
            error_message TEXT,
            used_previous_params INTEGER,
            reoptimized INTEGER,
            created_at TEXT,
            updated_at TEXT
        )
    """)

    # Cria índices para consultas rápidas
    SQLite.execute(db, """
        CREATE INDEX IF NOT EXISTS idx_date ON nss_curves(date)
    """)

    SQLite.execute(db, """
        CREATE INDEX IF NOT EXISTS idx_success ON nss_curves(success)
    """)

    return db
end

# ============================================================================
# Funções de Escrita
# ============================================================================

"""
    save_curve(db::SQLite.DB, date::Date, params::Vector{Float64},
               cost::Float64, n_bonds::Int, outliers_removed::Int;
               success::Bool=true, error_message::Union{String,Nothing}=nothing,
               used_previous_params::Bool=false, reoptimized::Bool=false)

Salva ou atualiza curva NSS no banco de dados.

# Argumentos
- `db`: Conexão ao banco SQLite
- `date`: Data de referência
- `params`: Vetor de 6 parâmetros NSS [β₀, β₁, β₂, β₃, τ₁, τ₂]
- `cost`: Custo de ajuste (erro médio)
- `n_bonds`: Número de títulos usados
- `outliers_removed`: Outliers removidos
- `success`: Se o ajuste foi bem-sucedido
- `error_message`: Mensagem de erro (opcional)
- `used_previous_params`: Se usou parâmetros anteriores
- `reoptimized`: Se foi reotimizado

# Exemplo
```julia
save_curve(db, Date(2024, 1, 2), params, 15.5, 25, 2)
```
"""
function save_curve(db::SQLite.DB, date::Date, params::Vector{Float64},
                   cost::Float64, n_bonds::Int, outliers_removed::Int;
                   success::Bool=true, error_message::Union{String,Nothing}=nothing,
                   used_previous_params::Bool=false, reoptimized::Bool=false)

    @assert length(params) == 6 "params deve ter 6 elementos [β₀, β₁, β₂, β₃, τ₁, τ₂]"

    date_str = Dates.format(date, "yyyy-mm-dd")
    timestamp = Dates.format(now(), "yyyy-mm-dd HH:MM:SS")

    # Verifica se já existe
    existing = DBInterface.execute(db, "SELECT date FROM nss_curves WHERE date = ?", [date_str])
    exists = !isempty(collect(existing))

    if exists
        # UPDATE
        SQLite.execute(db, """
            UPDATE nss_curves SET
                beta0 = ?, beta1 = ?, beta2 = ?, beta3 = ?, tau1 = ?, tau2 = ?,
                cost = ?, n_bonds = ?, outliers_removed = ?,
                success = ?, error_message = ?,
                used_previous_params = ?, reoptimized = ?,
                updated_at = ?
            WHERE date = ?
        """, [params[1], params[2], params[3], params[4], params[5], params[6],
              cost, n_bonds, outliers_removed,
              success, error_message,
              used_previous_params, reoptimized,
              timestamp, date_str])
    else
        # INSERT
        SQLite.execute(db, """
            INSERT INTO nss_curves (
                date, beta0, beta1, beta2, beta3, tau1, tau2,
                cost, n_bonds, outliers_removed,
                success, error_message,
                used_previous_params, reoptimized,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [date_str, params[1], params[2], params[3], params[4], params[5], params[6],
              cost, n_bonds, outliers_removed,
              success, error_message,
              used_previous_params, reoptimized,
              timestamp, timestamp])
    end

    return nothing
end

"""
    save_curve_failure(db::SQLite.DB, date::Date, error_message::String)

Salva registro de falha de ajuste.

# Exemplo
```julia
save_curve_failure(db, Date(2024, 1, 2), "Dados insuficientes")
```
"""
function save_curve_failure(db::SQLite.DB, date::Date, error_message::String)
    date_str = Dates.format(date, "yyyy-mm-dd")
    timestamp = Dates.format(now(), "yyyy-mm-dd HH:MM:SS")

    # Verifica se já existe
    existing = DBInterface.execute(db, "SELECT date FROM nss_curves WHERE date = ?", [date_str])
    exists = !isempty(collect(existing))

    if exists
        SQLite.execute(db, """
            UPDATE nss_curves SET
                success = 0,
                error_message = ?,
                updated_at = ?
            WHERE date = ?
        """, [error_message, timestamp, date_str])
    else
        SQLite.execute(db, """
            INSERT INTO nss_curves (
                date, success, error_message, created_at, updated_at
            ) VALUES (?, 0, ?, ?, ?)
        """, [date_str, error_message, timestamp, timestamp])
    end

    return nothing
end

# ============================================================================
# Funções de Leitura
# ============================================================================

"""
    load_curves(db::SQLite.DB, start_date::Date, end_date::Date) -> DataFrame

Carrega curvas NSS do banco para o intervalo especificado.

# Retorna
DataFrame com colunas:
- `date`, `beta0`, `beta1`, `beta2`, `beta3`, `tau1`, `tau2`
- `cost`, `n_bonds`, `outliers_removed`
- `success`, `error_message`, `used_previous_params`, `reoptimized`
- `created_at`, `updated_at`

# Exemplo
```julia
curves = load_curves(db, Date(2024, 1, 1), Date(2024, 12, 31))
```
"""
function load_curves(db::SQLite.DB, start_date::Date, end_date::Date)
    start_str = Dates.format(start_date, "yyyy-mm-dd")
    end_str = Dates.format(end_date, "yyyy-mm-dd")

    result = DBInterface.execute(db, """
        SELECT * FROM nss_curves
        WHERE date >= ? AND date <= ?
        ORDER BY date
    """, [start_str, end_str])

    df = DataFrame(result)

    # Converte coluna date para Date
    if !isempty(df)
        df.date = Date.(df.date)
    end

    return df
end

"""
    load_curve(db::SQLite.DB, date::Date) -> Union{NamedTuple, Nothing}

Carrega curva NSS para uma data específica.

# Retorna
- NamedTuple com parâmetros e metadados se existir
- `nothing` se não existir

# Exemplo
```julia
curve = load_curve(db, Date(2024, 1, 2))
if curve !== nothing
    println("β₀ = \$(curve.beta0)")
end
```
"""
function load_curve(db::SQLite.DB, date::Date)
    date_str = Dates.format(date, "yyyy-mm-dd")

    result = DBInterface.execute(db, """
        SELECT * FROM nss_curves WHERE date = ?
    """, [date_str])

    df = DataFrame(result)

    if isempty(df)
        return nothing
    end

    row = df[1, :]

    success_bool = !ismissing(row.success) && row.success == 1
    date_value = !ismissing(row.date) ? Date(row.date) : nothing

    return (
        date = date_value,
        params = success_bool ? [row.beta0, row.beta1, row.beta2, row.beta3, row.tau1, row.tau2] : nothing,
        cost = success_bool ? row.cost : nothing,
        n_bonds = !ismissing(row.n_bonds) ? row.n_bonds : 0,
        outliers_removed = !ismissing(row.outliers_removed) ? row.outliers_removed : 0,
        success = success_bool,
        error_message = !ismissing(row.error_message) ? row.error_message : nothing,
        used_previous_params = !ismissing(row.used_previous_params) && row.used_previous_params == 1,
        reoptimized = !ismissing(row.reoptimized) && row.reoptimized == 1,
        created_at = row.created_at,
        updated_at = row.updated_at
    )
end

"""
    curve_exists(db::SQLite.DB, date::Date) -> Bool

Verifica se curva já existe no banco.

# Exemplo
```julia
if !curve_exists(db, Date(2024, 1, 2))
    # Processar essa data
end
```
"""
function curve_exists(db::SQLite.DB, date::Date)
    date_str = Dates.format(date, "yyyy-mm-dd")

    result = DBInterface.execute(db, """
        SELECT COUNT(*) as count FROM nss_curves WHERE date = ?
    """, [date_str])

    for row in result
        return row.count > 0
    end

    return false
end

"""
    get_missing_dates(db::SQLite.DB, start_date::Date, end_date::Date;
                     only_business_days::Bool=true) -> Vector{Date}

Retorna lista de datas faltantes no banco (que precisam ser processadas).

# Argumentos
- `db`: Conexão ao banco
- `start_date`, `end_date`: Intervalo
- `only_business_days`: Se true, filtra apenas dias úteis (seg-sex)

# Exemplo
```julia
missing = get_missing_dates(db, Date(2024, 1, 1), Date(2024, 12, 31))
println("Faltam processar: \$(length(missing)) dias")
```
"""
function get_missing_dates(db::SQLite.DB, start_date::Date, end_date::Date;
                          only_business_days::Bool=true)
    # Gera todas as datas do intervalo
    all_dates = start_date:Day(1):end_date

    # Filtra apenas dias úteis se solicitado
    if only_business_days
        all_dates = filter(d -> dayofweek(d) ∉ [6, 7], collect(all_dates))
    else
        all_dates = collect(all_dates)
    end

    # Carrega datas existentes
    start_str = Dates.format(start_date, "yyyy-mm-dd")
    end_str = Dates.format(end_date, "yyyy-mm-dd")

    result = DBInterface.execute(db, """
        SELECT date FROM nss_curves
        WHERE date >= ? AND date <= ?
    """, [start_str, end_str])

    existing_dates = Set(Date(row.date) for row in result)

    # Retorna datas faltantes
    return filter(d -> d ∉ existing_dates, all_dates)
end

# ============================================================================
# Funções de Estatísticas
# ============================================================================

"""
    get_database_stats(db::SQLite.DB) -> NamedTuple

Retorna estatísticas do banco de dados.

# Retorna
NamedTuple com:
- `total_curves`: Total de registros
- `successful_curves`: Curvas com sucesso
- `failed_curves`: Curvas que falharam
- `date_range`: (primeira_data, última_data)
- `success_rate`: Taxa de sucesso (%)

# Exemplo
```julia
stats = get_database_stats(db)
println("Taxa de sucesso: \$(stats.success_rate)%")
```
"""
function get_database_stats(db::SQLite.DB)
    # Total
    result = DBInterface.execute(db, "SELECT COUNT(*) as count FROM nss_curves")
    total = 0
    for row in result
        total = !ismissing(row.count) ? row.count : 0
    end

    if total == 0
        return (
            total_curves = 0,
            successful_curves = 0,
            failed_curves = 0,
            date_range = (nothing, nothing),
            success_rate = 0.0
        )
    end

    # Sucessos
    result = DBInterface.execute(db, "SELECT COUNT(*) as count FROM nss_curves WHERE success = 1")
    successful = 0
    for row in result
        successful = !ismissing(row.count) ? row.count : 0
    end

    # Range de datas
    result = DBInterface.execute(db, "SELECT MIN(date) as min_date, MAX(date) as max_date FROM nss_curves")
    min_date = nothing
    max_date = nothing
    for row in result
        min_date = !ismissing(row.min_date) ? Date(row.min_date) : nothing
        max_date = !ismissing(row.max_date) ? Date(row.max_date) : nothing
    end

    return (
        total_curves = total,
        successful_curves = successful,
        failed_curves = total - successful,
        date_range = (min_date, max_date),
        success_rate = round(100 * successful / total, digits=1)
    )
end

# ============================================================================
# Consulta de Taxas (múltiplos métodos via multiple dispatch)
# ============================================================================

"""
    get_rate(db::SQLite.DB, date::Date, maturity::Real) -> Union{Float64, Nothing}

Retorna taxa NSS para uma data e prazo específicos.

# Argumentos
- `db`: Conexão ao banco SQLite
- `date`: Data de referência
- `maturity`: Prazo em anos (ex: 0.5 para 6 meses, 1.0 para 1 ano)

# Retorna
- Taxa (Float64) se curva existir e for bem-sucedida
- `nothing` se curva não existir ou falhou

# Exemplo
```julia
db = init_database("historical_curves.db")

# Taxa para 1 ano em 2024-06-15
rate = get_rate(db, Date(2024, 6, 15), 1.0)  # → 0.1025 ou nothing

if rate !== nothing
    println("DI 1Y: \$(round(rate * 100, digits=2))%")
end
```
"""
function get_rate(db::SQLite.DB, date::Date, maturity::Real)
    curve = load_curve(db, date)

    if curve === nothing || !curve.success || curve.params === nothing
        return nothing
    end

    return nss_rate(maturity, curve.params)
end

"""
    get_rate(db::SQLite.DB, date::Date, maturities::Vector{<:Real}) -> Vector{Union{Float64, Nothing}}

Retorna taxas NSS para múltiplos prazos em uma data específica.

# Argumentos
- `db`: Conexão ao banco SQLite
- `date`: Data de referência
- `maturities`: Vetor de prazos em anos

# Retorna
Vetor com taxas (ou `nothing` para cada prazo se curva não existir)

# Exemplo
```julia
db = init_database("historical_curves.db")

# Múltiplos prazos na mesma data
rates = get_rate(db, Date(2024, 6, 15), [0.5, 1.0, 2.0, 5.0, 10.0])

# rates = [0.0985, 0.1025, 0.1075, 0.1150, 0.1200] ou [nothing, ...]
```
"""
function get_rate(db::SQLite.DB, date::Date, maturities::Vector{<:Real})
    curve = load_curve(db, date)

    if curve === nothing || !curve.success || curve.params === nothing
        return fill(nothing, length(maturities))
    end

    return [nss_rate(m, curve.params) for m in maturities]
end

"""
    get_rate(db::SQLite.DB, start_date::Date, end_date::Date, maturity::Real) -> DataFrame

Retorna série temporal de taxas NSS para um prazo específico.

# Argumentos
- `db`: Conexão ao banco SQLite
- `start_date`: Data inicial
- `end_date`: Data final
- `maturity`: Prazo em anos

# Retorna
DataFrame com colunas:
- `date`: Data
- `rate`: Taxa NSS (ou `missing` se não disponível)

# Exemplo
```julia
db = init_database("historical_curves.db")

# Série temporal: DI 1 ano ao longo de 2024
series = get_rate(db, Date(2024, 1, 1), Date(2024, 12, 31), 1.0)

# Filtrar apenas valores válidos
valid_series = dropmissing(series)

# Plotar
using Plots
plot(valid_series.date, valid_series.rate .* 100,
     xlabel="Data", ylabel="Taxa (%)", title="DI 1 ano - 2024")
```
"""
function get_rate(db::SQLite.DB, start_date::Date, end_date::Date, maturity::Real)
    # Carrega todas as curvas do intervalo
    curves_df = load_curves(db, start_date, end_date)

    if isempty(curves_df)
        return DataFrame(date=Date[], rate=Union{Float64,Missing}[])
    end

    # Calcula taxa para cada curva
    rates = Vector{Union{Float64,Missing}}(undef, nrow(curves_df))

    for (i, row) in enumerate(eachrow(curves_df))
        if row.success == 1 && !ismissing(row.beta0)
            params = [row.beta0, row.beta1, row.beta2, row.beta3, row.tau1, row.tau2]
            rates[i] = nss_rate(maturity, params)
        else
            rates[i] = missing
        end
    end

    return DataFrame(date=curves_df.date, rate=rates)
end

"""
    get_rate(db::SQLite.DB, dates::Vector{Date}, maturity::Real) -> DataFrame

Retorna taxas NSS para datas específicas e um prazo.

# Argumentos
- `db`: Conexão ao banco SQLite
- `dates`: Vetor de datas
- `maturity`: Prazo em anos

# Retorna
DataFrame com colunas:
- `date`: Data
- `rate`: Taxa NSS (ou `missing` se não disponível)

# Exemplo
```julia
db = init_database("historical_curves.db")

# Datas específicas
dates = [Date(2024, 1, 2), Date(2024, 4, 1), Date(2024, 7, 1), Date(2024, 10, 1)]
rates = get_rate(db, dates, 5.0)  # DI 5 anos

println(rates)
```
"""
function get_rate(db::SQLite.DB, dates::Vector{Date}, maturity::Real)
    rates = Vector{Union{Float64,Missing}}(undef, length(dates))

    for (i, date) in enumerate(dates)
        curve = load_curve(db, date)

        if curve !== nothing && curve.success && curve.params !== nothing
            rates[i] = nss_rate(maturity, curve.params)
        else
            rates[i] = missing
        end
    end

    return DataFrame(date=dates, rate=rates)
end
