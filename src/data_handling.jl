"""
data_handling.jl - Manipulação de dados e gerenciamento de configuração

Este módulo gerencia todas as operações de I/O de dados incluindo:
- Carregamento de dados de negociação de títulos do BACEN
- Gerenciamento de arquivos de configuração (formato TOML)
- Geração de fluxos de caixa
- Filtragem e validação de dados
- Gerenciamento inteligente de cache com validação por watermark
"""

using HTTP, DataFrames, ZipFile, CSV, Dates, TOML, JSON, BusinessDays

# Cache para dados de títulos mensais (melhora performance)
const BOND_DATA_CACHE = Dict{String, DataFrame}()

"""
    load_configuration(config_file::String="config.toml") -> Dict{String, Any}

Load configuration from TOML file.

Parameters:
- config_file: Path to configuration file (default: "config.toml")

Returns:
- Dictionary with configuration parameters
"""
function load_configuration(config_file::String="config.toml")
    if !isfile(config_file)
        error("Configuration file $config_file not found!")
    end
    
    config = TOML.parsefile(config_file)
    
    # Validate required sections
    required_sections = ["pso", "optimization", "outlier_detection", "validation"]
    for section in required_sections
        if !haskey(config, section)
            error("Missing required configuration section: $section")
        end
    end
    
    return config
end

"""
    save_optimal_configuration(config::Dict{String, Any}, 
                              performance::Dict{String, Any},
                              output_file::String="optimal_config.toml")

Save optimal configuration and performance metrics to TOML file.

Parameters:
- config: Configuration dictionary
- performance: Performance metrics dictionary  
- output_file: Output file path (default: "optimal_config.toml")
"""
function save_optimal_configuration(config::Dict{String, Any}, 
                                   performance::Dict{String, Any},
                                   output_file::String="optimal_config.toml")
    
    output_data = Dict{String, Any}(
        "optimal_config" => config,
        "performance_metrics" => performance,
        "metadata" => Dict{String, Any}(
            "generated_at" => string(now()),
            "methodology" => "PSO_plus_LM_hybrid_bayesian_optimization",
            "cv_method" => "cross_regime_walk_forward_30days"
        )
    )
    
    open(output_file, "w") do io
        TOML.print(io, output_data)
    end
    
    println("✅ Optimal configuration saved to: $output_file")
end

"""
    get_business_dates(start_date::Date, end_date::Date; calendar=BusinessDays.BRSettlement()) -> Vector{Date}

Generate vector of business dates between start and end dates (inclusive), considering Brazilian holidays.

Uses BusinessDays.jl with Brazilian calendar (BRSettlement) to correctly exclude:
- Weekends (Saturday and Sunday)
- Brazilian national holidays (Carnaval, Corpus Christi, Independence Day, etc.)

Parameters:
- start_date: First date (inclusive)
- end_date: Last date (inclusive)
- calendar: Holiday calendar (default: Brazilian calendar with national holidays)

Returns:
- Vector{Date} with all business days in the range

Example:
```julia
# Get all business days in June 2025 (excluding Brazilian holidays)
dates = get_business_dates(Date(2025, 6, 1), Date(2025, 6, 30))
```
"""
function get_business_dates(start_date::Date, end_date::Date; calendar=BusinessDays.BRSettlement())
    return BusinessDays.listbdays(calendar, start_date, end_date)
end

"""
    get_pso_bounds(config::Dict) -> Tuple{Vector{Float64}, Vector{Float64}}

Get PSO parameter bounds from configuration file with sensible defaults.

This is the single source of truth for NSS parameter bounds:
[β₀, β₁, β₂, β₃, τ₁, τ₂]

Parameters:
- config: Configuration dictionary (typically from config.toml)

Returns:
- Tuple of (lower_bounds, upper_bounds) vectors

Default bounds if not specified in config:
- β₀ (level): [0.01, 0.30] - long-term rate
- β₁ (slope): [-0.25, 0.25] - short-term component
- β₂ (curvature): [-0.30, 0.30] - medium-term component
- β₃ (Svensson): [-0.20, 0.20] - additional curvature
- τ₁ (decay 1): [0.5, 20.0] - first time constant
- τ₂ (decay 2): [2.0, 50.0] - second time constant

Example:
```julia
config = TOML.parsefile("config.toml")
lower, upper = get_pso_bounds(config)
```
"""
function get_pso_bounds(config::Dict)
    pso_config = get(config, "pso", Dict())

    # Sensible defaults for Brazilian yield curves
    default_lower = [0.01, -0.25, -0.30, -0.20, 0.5, 2.0]
    default_upper = [0.30, 0.25, 0.30, 0.20, 20.0, 50.0]

    lower_bounds = get(pso_config, "lower_bounds", default_lower)
    upper_bounds = get(pso_config, "upper_bounds", default_upper)

    return (lower_bounds, upper_bounds)
end

# Helper functions for BACEN data processing

function find_csv_file(zip_reader)
    for file in zip_reader.files
        if endswith(uppercase(file.name), ".CSV")
            return file
        end
    end
    return nothing
end

function read_bacen_csv(csv_file)
    df = CSV.read(csv_file, DataFrame; 
                  delim=';', decimal=',', dateformat="d/m/y", missingstring="",
                  types=Dict("DATA MOV" => Date, "VENCIMENTO" => Date, "PU MED" => Float64))
    
    # Verifica se a coluna ISIN existe
    @assert any(occursin("ISIN", uppercase(string(col))) for col in names(df)) "Coluna ISIN não encontrada"
    
    return df
end

"""
    is_cache_valid(year_month::String, target_date::Date, cache_config::Dict{String,Any}) -> Bool

Verifica se o arquivo em cache é válido para a data solicitada.

Parâmetros:
- year_month: String ano-mês (YYYYMM)
- target_date: Data sendo solicitada
- cache_config: Configuração de cache do config.toml

Retorna:
- true se o cache é válido e pode ser usado, false caso contrário

Regras de validação:
- Meses passados: sempre válido (dados imutáveis)
- Mês corrente: inválido se target_date > última_data_no_cache ou cache muito antigo
"""
function is_cache_valid(year_month::String, target_date::Date, cache_config::Dict{String,Any})::Bool
    metadata = load_cache_metadata()

    # Sem metadados para este mês
    !haskey(metadata, year_month) && return false

    cache_info = metadata[year_month]

    # Arquivo não existe mais
    !isfile(cache_info.file_path) && return false

    # Verifica se é o mês corrente
    current_ym = Dates.format(today(), "yyyymm")

    # Mês passado → validar se foi baixado APÓS o mês terminar
    if year_month < current_ym
        # Determinar o último dia do mês
        year = parse(Int, year_month[1:4])
        month = parse(Int, year_month[5:6])
        last_day_of_month = Date(year, month) + Month(1) - Day(1)

        # Cache baixado DURANTE o mês → incompleto → inválido
        if Date(cache_info.downloaded_at) <= last_day_of_month
            return false
        end

        # Cache baixado APÓS o mês terminar → completo → válido
        return true
    end

    # Mês corrente → valida por data e idade
    if year_month == current_ym
        # Precisa de dados mais novos do que o cache tem
        if target_date > cache_info.last_date
            return false
        end

        # Cache muito antigo (obsoleto)
        hours_since_download = (now() - cache_info.downloaded_at).value / 1000 / 3600
        max_age_hours = get(cache_config, "max_age_hours", 24)
        if hours_since_download > max_age_hours
            return false
        end
    end

    return true
end

function download_zip_file(url::String, output_path::String, target_date::Date; force::Bool=false)
    raw_dir = "raw"
    !isdir(raw_dir) && mkdir(raw_dir)

    full_path = joinpath(raw_dir, output_path)

    # Extrai ano_mês do nome do arquivo
    m = match(r"bacen_(\d{6})\.zip", output_path)
    year_month = !isnothing(m) ? m.captures[1] : nothing

    # Carrega configuração de cache usando ConfigService
    cache_config = Dict{String, Any}()
    if isfile("config.toml")
        try
            config_service = default_config()
            cache_section = get(get_raw_config(config_service), "cache", Dict{String, Any}())
            cache_config = Dict{String, Any}(cache_section)
        catch e
            @warn "Falha ao carregar configuração de cache: $e"
        end
    end

    # Verifica se cache é válido (exceto se forçar redownload)
    # TEMPORÁRIO: Desabilitando validação complexa de cache devido a problema de world age
    # TODO: Resolver problema de world age com is_cache_valid
    if !force && !isnothing(year_month) && isfile(full_path)
        # Para meses passados, sempre usa cache
        # Para mês corrente, força redownload se configurado
        current_ym = Dates.format(today(), "yyyymm")
        force_redownload = get(cache_config, "force_redownload", false)

        if year_month < current_ym || !force_redownload
            metadata = load_cache_metadata()
            last_date = haskey(metadata, year_month) ? metadata[year_month].last_date : Date(0)
            println("📁 Usando arquivo local existente: $full_path (dados até $last_date)")
            return ZipFile.Reader(full_path)
        end
    end

    # Download necessário
    if isfile(full_path)
        println("🔄 Arquivo existe mas está desatualizado, baixando nova versão...")
    else
        println("⬇️  Baixando: $url")
    end

    response = HTTP.get(url; retries=2, readtimeout=60)

    write(full_path, response.body)

    # Atualiza metadados do cache
    if !isnothing(year_month)
        try
            last_date = extract_last_date_from_zip(full_path)
            metadata = load_cache_metadata()
            metadata[year_month] = CacheMetadata(
                now(),
                last_date,
                filesize(full_path),
                full_path
            )
            save_cache_metadata(metadata)
            println("💾 Arquivo salvo: $full_path (dados até $last_date)")
        catch e
            @warn "Falha ao atualizar metadados do cache: $e"
            println("💾 Arquivo salvo em: $full_path")
        end
    else
        println("💾 Arquivo salvo em: $full_path")
    end

    return ZipFile.Reader(IOBuffer(response.body))
end

function process_month_data(year_month::String, base_url::String, target_date::Date, save_zip::Bool=false)
    url = base_url * year_month * ".zip"

    # Baixa e processa o arquivo zip
    output_file = "bacen_$(year_month).zip"
    zip_reader = download_zip_file(url, output_file, target_date)
    # Encontra e lê o arquivo CSV
    csv_file = find_csv_file(zip_reader)
    @assert !isnothing(csv_file) "Nenhum arquivo CSV encontrado no ZIP para $year_month"

    println("Lendo CSV: $(csv_file.name)")
    df = read_bacen_csv(csv_file)
    close(zip_reader)
    return df
end

function clean_bacen_data(df::DataFrame, start_date::Date, end_date::Date)
    isempty(df) && return DataFrame()
    
    # Renomear colunas
    rename_map = Dict(
        "DATA MOV" => "date",
        "SIGLA" => "bond_code",
        "CODIGO" => "codigo",
        "CODIGO ISIN" => "isin",
        "VENCIMENTO" => "maturity_date",
        "PU MED" => "avg_price",
        "QUANT NEGOCIADA" => "quantity_traded"
    )
    rename!(df, Dict(Symbol(k) => Symbol(v) for (k, v) in rename_map if hasproperty(df, Symbol(k))))
    
    # Verificar colunas essenciais
    essential_cols = [:date, :bond_code, :maturity_date, :avg_price]
    @assert all(col -> hasproperty(df, col), essential_cols) "Colunas essenciais faltando após renomeação"
    
    # Selecionar colunas, remover missings e filtrar por data
    select_cols = [:date, :bond_code, :codigo, :isin, :maturity_date, :avg_price, :quantity_traded]
    available_cols = intersect(select_cols, propertynames(df))
    select!(df, available_cols)
    dropmissing!(df, intersect(essential_cols, propertynames(df)))
    
    # Filtrar pelo intervalo de datas exato [start_date, end_date]
    eltype(df.date) <: Date && filter!(row -> start_date <= row.date <= end_date, df)
    
    return df
end

"""
    load_bacen_data(start_date::Date, end_date::Date) -> DataFrame

Load BACEN bond trading data for the specified date range.

Parameters:
- start_date: Start date for data loading
- end_date: End date for data loading

Returns:
- DataFrame with bond trading data
"""
function load_bacen_data(target_date::Date, end_date::Date; save_zip::Bool=false)
    base_url = "https://www4.bcb.gov.br/pom/demab/negociacoes/download/NegT"
    year_month = Dates.format(target_date, "yyyymm")

    # 1. Tenta obter os dados mensais do cache
    if haskey(BOND_DATA_CACHE, year_month)
        # @info "Cache hit for bond data: $year_month"
        monthly_df = BOND_DATA_CACHE[year_month]
    else
        # 2. Se não está no cache, carrega do arquivo e salva no cache
        # @info "Cache miss for bond data: $year_month. Loading from disk..."
        try
            monthly_df = process_month_data(year_month, base_url, target_date, save_zip)
            if !isempty(monthly_df)
                BOND_DATA_CACHE[year_month] = monthly_df
            end
        catch e
            @warn "Falha ao carregar dados para $year_month: $e"
            return DataFrame() # Retorna DF vazio em caso de falha
        end
    end

    # 3. Filtra o DataFrame do mês para a data específica
    if isempty(monthly_df)
        return DataFrame()
    end
    
    # Renomeia colunas e limpa os dados
    clean_df = clean_bacen_data(copy(monthly_df), target_date, end_date) # Usa cópia para não alterar o cache
    
    # Filtra pelo dia exato [target_date]
    filter!(row -> row.date == target_date, clean_df)

    return clean_df
end

"""
    generate_cash_flows_with_quantity(df::DataFrame, reference_date::Date) -> Tuple

Generate cash flows with quantity information for bond optimization.

Parameters:
- df: DataFrame with bond data
- reference_date: Reference date for cash flow calculations

Returns:
- Tuple of (cash_flows, quantities, bond_info)
"""
function generate_cash_flows_with_quantity(df::DataFrame, reference_date::Date)
    cash_flows = Tuple{Float64, Vector{Tuple{Date, Float64}}}[]
    quantities = Float64[]
    bond_info = Dict{String, Any}[]
    
    for row in eachrow(df)
        # Extract bond information
        market_price = row.avg_price
        maturity_date = row.maturity_date
        quantity = hasproperty(df, :quantity_traded) ? row.quantity_traded : DEFAULT_BOND_QUANTITY
        bond_code = row.bond_code

        # Generate cash flows based on bond type
        if bond_code == "LTN"  # Zero-coupon bond
            cash_flow = [(maturity_date, LTN_FACE_VALUE)]  # Face value at maturity
        elseif bond_code == "NTN-F"  # Fixed-rate coupon bond
            # Semi-annual coupons at 10% p.a.
            cash_flow = []
            current_date = reference_date

            # Generate regular coupon payments (excluding maturity date)
            while current_date < maturity_date
                # Add 6 months for next coupon
                current_date = current_date + Month(6)
                if current_date < maturity_date
                    # Regular coupon payment
                    push!(cash_flow, (current_date, NTNF_COUPON_VALUE))  # 5% semi-annual coupon
                end
            end

            # Always add final payment at maturity: coupon + principal
            push!(cash_flow, (maturity_date, NTNF_MATURITY_VALUE))  # Final coupon + face value
        else
            continue  # Skip unknown bond types
        end
        
        # Add to results
        push!(cash_flows, (market_price, cash_flow))
        push!(quantities, quantity)
        push!(bond_info, Dict(
            "bond_code" => bond_code,
            "maturity_date" => maturity_date,
            "codigo" => hasproperty(df, :codigo) ? row.codigo : missing
        ))
    end
    
    return cash_flows, quantities, bond_info
end

# ============================================================================
# Gerenciamento Inteligente de Cache com Watermark
# ============================================================================

"""
Estrutura de metadados do cache para rastrear arquivos baixados e sua validade.
"""
struct CacheMetadata
    downloaded_at::DateTime
    last_date::Date
    file_size::Int64
    file_path::String
end

# Converte CacheMetadata para Dict para serialização JSON
function Base.Dict(cm::CacheMetadata)
    return Dict(
        "downloaded_at" => Dates.format(cm.downloaded_at, "yyyy-mm-ddTHH:MM:SS"),
        "last_date" => Dates.format(cm.last_date, "yyyy-mm-dd"),
        "file_size" => cm.file_size,
        "file_path" => cm.file_path
    )
end

# Converte Dict para CacheMetadata
function CacheMetadata(d::Dict)
    return CacheMetadata(
        DateTime(d["downloaded_at"], "yyyy-mm-ddTHH:MM:SS"),
        Date(d["last_date"], "yyyy-mm-dd"),
        Int64(d["file_size"]),
        d["file_path"]
    )
end

const CACHE_METADATA_FILE = "raw/.cache_metadata.json"

"""
    load_cache_metadata() -> Dict{String, CacheMetadata}

Carrega metadados do cache a partir de arquivo JSON.

Retorna:
- Dicionário mapeando ano_mês (YYYYMM) para CacheMetadata
"""
function load_cache_metadata()::Dict{String, CacheMetadata}
    !isfile(CACHE_METADATA_FILE) && return Dict{String, CacheMetadata}()

    try
        data = JSON.parsefile(CACHE_METADATA_FILE)
        return Dict(k => CacheMetadata(v) for (k, v) in data)
    catch e
        @warn "Falha ao carregar metadados do cache: $e. Iniciando com cache vazio."
        return Dict{String, CacheMetadata}()
    end
end

"""
    save_cache_metadata(metadata::Dict{String, CacheMetadata})

Salva metadados do cache em arquivo JSON.

Parâmetros:
- metadata: Dicionário mapeando ano_mês para CacheMetadata
"""
function save_cache_metadata(metadata::Dict{String, CacheMetadata})
    raw_dir = "raw"
    !isdir(raw_dir) && mkdir(raw_dir)

    data = Dict(k => Dict(v) for (k, v) in metadata)

    open(CACHE_METADATA_FILE, "w") do io
        JSON.print(io, data, 2)  # Impressão formatada com indentação
    end
end

"""
    extract_last_date_from_zip(zip_path::String) -> Date

Extrai a última data (mais recente) de um arquivo ZIP de dados do BACEN.

Parâmetros:
- zip_path: Caminho para o arquivo ZIP

Retorna:
- A data mais recente encontrada nos dados
"""
function extract_last_date_from_zip(zip_path::String)::Date
    zip_reader = ZipFile.Reader(zip_path)

    try
        csv_file = find_csv_file(zip_reader)
        @assert !isnothing(csv_file) "Nenhum arquivo CSV encontrado no ZIP"

        # Lê apenas a coluna de data de forma eficiente
        df = read_bacen_csv(csv_file)

        close(zip_reader)

        # Encontra a data máxima
        return maximum(df[!, Symbol("DATA MOV")])
    catch e
        close(zip_reader)
        @warn "Falha ao extrair última data de $zip_path: $e"
        # Retorna estimativa conservadora (primeiro dia do mês)
        year_month = match(r"bacen_(\d{6})\.zip", basename(zip_path))
        if !isnothing(year_month)
            ym = year_month.captures[1]
            return Date(ym, "yyyymm")
        end
        rethrow(e)
    end
end


"""
    clear_cache(; year_month::Union{String,Nothing}=nothing)

Limpa metadados do cache e opcionalmente remove arquivos em cache.

Parâmetros:
- year_month: Mês específico para limpar (YYYYMM), ou nothing para limpar tudo

Exemplos:
```julia
clear_cache()  # Limpa todo o cache
clear_cache(year_month="202405")  # Limpa apenas maio de 2024
```
"""
function clear_cache(; year_month::Union{String,Nothing}=nothing)
    metadata = load_cache_metadata()

    if isnothing(year_month)
        # Limpa todo o cache
        for (ym, info) in metadata
            if isfile(info.file_path)
                rm(info.file_path, force=true)
                println("🗑️  Removido: $(info.file_path)")
            end
        end

        # Remove arquivo de metadados
        if isfile(CACHE_METADATA_FILE)
            rm(CACHE_METADATA_FILE, force=true)
        end

        println("✅ Todo o cache foi limpo")
    else
        # Limpa mês específico
        if haskey(metadata, year_month)
            info = metadata[year_month]
            if isfile(info.file_path)
                rm(info.file_path, force=true)
                println("🗑️  Removido: $(info.file_path)")
            end

            delete!(metadata, year_month)
            save_cache_metadata(metadata)
            println("✅ Cache para $year_month limpo")
        else
            println("⚠️  Nenhum cache encontrado para $year_month")
        end
    end
end

"""
    migrate_existing_cache()

Escaneia arquivos ZIP existentes no diretório raw/ e gera metadados para eles.
Útil para migrar do sistema de cache antigo para o novo sistema com watermark.
"""
function migrate_existing_cache()
    raw_dir = "raw"
    !isdir(raw_dir) && return

    zip_files = filter(f -> endswith(f, ".zip") && startswith(basename(f), "bacen_"), readdir(raw_dir, join=true))
    isempty(zip_files) && return

    println("🔄 Migrando $(length(zip_files)) arquivos de cache existentes...")

    metadata = load_cache_metadata()
    migrated = 0

    for zip_path in zip_files
        # Extrai ano_mês do nome do arquivo
        m = match(r"bacen_(\d{6})\.zip", basename(zip_path))
        isnothing(m) && continue

        year_month = m.captures[1]

        # Pula se já tem metadados
        haskey(metadata, year_month) && continue

        try
            last_date = extract_last_date_from_zip(zip_path)
            # Converte mtime Unix para DateTime
            # stat().mtime retorna Float64 representando segundos desde Unix epoch (1970-01-01)
            # Multiplicamos por 1000 para converter para milissegundos, que é o que DateTime usa internamente
            mtime_ms = round(Int64, stat(zip_path).mtime * 1000)
            # Epoch Unix em DateTime Julia é DateTime(1970,1,1)
            epoch = DateTime(1970,1,1)
            mtime_datetime = epoch + Millisecond(mtime_ms)

            metadata[year_month] = CacheMetadata(
                mtime_datetime,  # Usa tempo de modificação do arquivo
                last_date,
                filesize(zip_path),
                zip_path
            )
            migrated += 1
            println("  ✅ $year_month → dados até $last_date")
        catch e
            @warn "Falha ao migrar $zip_path: $e"
        end
    end

    if migrated > 0
        save_cache_metadata(metadata)
        println("✅ $migrated arquivos de cache migrados")
    else
        println("ℹ️  Nenhum arquivo novo para migrar")
    end
end