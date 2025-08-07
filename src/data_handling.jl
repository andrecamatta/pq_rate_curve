"""
data_handling.jl - Data manipulation and configuration management

This module handles all data I/O operations including:
- Loading BACEN bond trading data
- Configuration file management (TOML format)
- Cash flow generation
- Data filtering and validation
"""

using HTTP, DataFrames, ZipFile, CSV, Dates, TOML

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

function download_zip_file(url::String, output_path::String)
    raw_dir = "raw"
    !isdir(raw_dir) && mkdir(raw_dir)
    
    full_path = joinpath(raw_dir, output_path)
    
    # Verifica se arquivo já existe localmente
    if isfile(full_path)
        println("📁 Usando arquivo local existente: $full_path")
        return ZipFile.Reader(full_path)
    end
    
    # Se não existe, faz download
    println("⬇️  Baixando: $url")
    response = HTTP.get(url; retries=2, readtimeout=60)
    
    write(full_path, response.body)
    println("💾 Arquivo salvo em: $full_path")
    
    return ZipFile.Reader(IOBuffer(response.body))
end

function process_month_data(year_month::String, base_url::String, save_zip::Bool=false)
    url = base_url * year_month * ".zip"
    
    # Baixa e processa o arquivo zip
    output_file = "bacen_$(year_month).zip"
    zip_reader = download_zip_file(url, output_file)
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
            monthly_df = process_month_data(year_month, base_url, save_zip)
            if !isempty(monthly_df)
                BOND_DATA_CACHE[year_month] = monthly_df
            end
        catch e
            @warn "Failed to load data for $year_month: $e"
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
        quantity = hasproperty(df, :quantity_traded) ? row.quantity_traded : 1000.0
        bond_code = row.bond_code
        
        # Generate cash flows based on bond type
        if bond_code == "LTN"  # Zero-coupon bond
            cash_flow = [(maturity_date, 1000.0)]  # Face value at maturity
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
                    push!(cash_flow, (current_date, 50.0))  # 5% semi-annual coupon
                end
            end
            
            # Always add final payment at maturity: coupon + principal
            push!(cash_flow, (maturity_date, 1000.0 + 50.0))  # Final coupon + face value
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