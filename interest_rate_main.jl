include("interest_rate_lib.jl")
include("scraping_tesouro_direto.jl")
include("scraping_tesouro_via_bacen.jl")

using Plots
using DataFrames
using Dates
using Statistics: mean
using Plots: text, default, scatter, scatter!, savefig, annotate!, plot!, plot
using PrettyTables
using XLSX # Adicionado para salvar em Excel

# Configuração do tema do gráfico
ENV["GKSwstype"] = "100"
default(fontfamily="Computer Modern", legend=:topright, grid=true)

"""
    get_bacen_data(ref_date::String)
    
Obtém dados do BACEN para a data de referência especificada.
Retorna os dados filtrados e processados para a data.
"""
function get_bacen_data(ref_date::String)
    # Converte a data de referência para o formato Date
    ref_dt = to_date(ref_date)
    
    # Define o período de busca (apenas o mês que contém a data de referência)
    start_date = Date(year(ref_dt), month(ref_dt), 1)
    end_date = lastdayofmonth(ref_dt)
    
    println("Baixando dados do BACEN para o período $(start_date) a $(end_date)...")
    
    # Baixa os dados do BACEN
    bacen_df = load_bacen_data(start_date, end_date)
    
    # Filtra os dados pela data específica
    filtered_df = filter(row -> !ismissing(row.date) && Date(row.date) == ref_dt, bacen_df)
    
    println("Total de registros encontrados para a data: $(nrow(filtered_df))")
    
    # Se não encontrar registros para a data, retorna erro
    if isempty(filtered_df)
        error("Dados não disponíveis para a data $(ref_date) no BACEN.")
    end
    
    # Verifica especificamente os títulos prefixados (LTN e NTN-F)
    prefixed_df = filter(row -> 
        !ismissing(row.bond_code) && 
        (row.bond_code == "LTN" || row.bond_code == "NTN-F"), 
        filtered_df)
    
    println("Total de títulos prefixados: $(nrow(prefixed_df))")
    
    # Prepara o DataFrame no formato esperado
    result_df = DataFrame(
        "Data Base" => String[],
        "Tipo Titulo" => String[],
        "Data Vencimento" => String[],
        "PU_BACEN_AVG" => Float64[]
    )
    
    for row in eachrow(prefixed_df)
        if !ismissing(row.bond_code) && !ismissing(row.avg_price) && !ismissing(row.maturity_date)
            # Converte as datas para o formato esperado
            data_base = format_date(row.date)
            
            # Se a data de vencimento já é um Date, formata. Caso contrário, usa como está.
            data_venc = if row.maturity_date isa Date
                format_date(row.maturity_date)
            else
                String(row.maturity_date)
            end
            
            # Define o tipo de título de forma compatível com o formato do Tesouro Direto
            tipo = if row.bond_code == "LTN" 
                "Tesouro Prefixado"
            else 
                "Tesouro Prefixado com Juros Semestrais"
            end
            
            # Adiciona a linha ao DataFrame resultante
            if !ismissing(row.avg_price)
                push!(result_df, (data_base, tipo, data_venc, row.avg_price))
            end
        end
    end
    
    # Verifica se temos dados suficientes
    if nrow(result_df) < 2
        error("Insuficientes títulos prefixados encontrados na data $(ref_date). São necessários pelo menos 2 títulos.")
    end
    
    println("Registros válidos processados: $(nrow(result_df))")
    
    return result_df
end

"""
    prepare_bonds(df, ref_date, source)
    
Prepara os objetos BRLBond a partir do DataFrame, usando a fonte especificada
"""
function prepare_bonds(df, ref_date::String, source::DataSource)
    # Para o Tesouro Direto, precisamos filtrar por data base
    filtered_df = if source == tesouro_direto_manha
        # Filtra por data base
        data_df = df[df[!, "Data Base"] .== ref_date, :]
        
        # Se não encontrar dados para a data específica, informa e interrompe
        if isempty(data_df)
            error("Dados não disponíveis para a data $ref_date.")
        end
        
        # Filtra somente prefixados
        data_df[occursin.("Prefixado", data_df[!, "Tipo Titulo"]), :]
    else
        # Para BACEN, os dados já foram filtrados na função get_bacen_data
        df
    end
    
    # Verifica se há títulos após a filtragem
    if isempty(filtered_df)
        error("Nenhum título prefixado encontrado para a data $ref_date.")
    end
    
    # Cria os objetos BRLBond
    bonds = BRLBond[]
    
    for row in eachrow(filtered_df)
        try
            pu = if source == tesouro_direto_manha
                # Média dos PUs de compra e venda
                pu_values = [parse(Float64, replace(string(row[k]), "," => ".")) 
                            for k in ["PU Compra Manha", "PU Venda Manha"]]
                mean(pu_values)
            elseif source == bacen_avg_price
                # PU médio do BACEN
                row["PU_BACEN_AVG"]
            else
                error("Fonte desconhecida")
            end
            
            # Cria o objeto BRLBond
            bond = BRLBond(
                row["Tipo Titulo"],
                row["Data Vencimento"],
                pu,
                has_coupon(row["Tipo Titulo"])
            )
            
            push!(bonds, bond)
        catch e
            @warn "Erro ao processar título $(row["Tipo Titulo"]) com vencimento $(row["Data Vencimento"]): $e"
        end
    end
    
    @assert !isempty(bonds) "Nenhum título válido encontrado após processamento"
    bonds
end

"""
    plot_yield_curve(complete_curve, ref_date, source; output_file)
    
Plota a curva de juros com títulos específicos para a fonte
"""
function plot_yield_curve(
    complete_curve::Dict{Date, Float64}, 
    ref_date::String, 
    source::DataSource;
    output_file::Union{String,Nothing}=nothing
)
    @assert !isempty(complete_curve) "Curva vazia, impossível plotar"
    
    # Prepara dados para plotagem
    dates = sort(collect(keys(complete_curve)))
    rates = [complete_curve[d] * 100 for d in dates] # Taxa em percentual
    years = [yearfrac(ref_date, format_date(d)) for d in dates]
    
    # Limites do gráfico
    min_rate, max_rate = extrema(rates)
    min_years, max_years = extrema(years)
    max_year = ceil(max_years)
    
    # Nome da fonte para título
    source_name = source == tesouro_direto_manha ? "Tesouro Direto (Manhã)" : "BACEN (PU Médio)"
    
    # Cria o plot
    p = plot(years, rates,
        xlabel="Anos",
        ylabel="Taxa (%)",
        title="Curva de Juros Pré ($source_name) - Ref: $ref_date",
        linewidth=2,
        color=:blue,
        size=(1600, 600),
        margin=20Plots.mm,
        ylims=(min_rate * 0.9, max_rate * 1.1),
        xlims=(min_years, max_year),
        legend=false
    )
    
    # Configura grid e ticks
    start_year = ceil(min_years * 2) / 2  # Próximo múltiplo de 0.5
    xticks_values = start_year:0.5:max_year
    plot!(p, xticks=xticks_values, grid=:x, gridstyle=:dot, gridalpha=0.3)
    
    # Anota taxas em pontos específicos
    ref_dt = to_date(ref_date)
    curve = create_curve(complete_curve, ref_dt)
    
    for year_mark in xticks_values
        try
            target_days = round(Int, year_mark * DAYS_IN_YEAR)
            rate = InterestRates.zero_rate(curve, target_days) * 100
            annotate!(p, year_mark, rate, text("$(round(rate, digits=2))%", 8, :bottom, :black))
        catch
            # Ignora pontos fora do range da curva
        end
    end
    
    # Salva e retorna
    output_file !== nothing && savefig(p, output_file)
    p
end

"""
    main(; ref_date, output_excel_file)

Função principal para construção e comparação das curvas
"""
function main(;
    ref_date::String=format_date(today() - Day(1)),
    output_excel_file::Union{String,Nothing}=nothing # Adicionado parâmetro para nome do arquivo Excel
)
    println("Gerando curvas de juros para $ref_date usando fontes: Tesouro Direto e BACEN")
    
    # Carrega dados do Tesouro Direto
    println("\n=== Carregando dados do Tesouro Direto ===")
    tesouro_df = load_treasury_data()
    tesouro_bonds = prepare_bonds(tesouro_df, ref_date, tesouro_direto_manha)
    println("Processando $(length(tesouro_bonds)) títulos do Tesouro Direto...")

    # Imprime os pontos usados para a curva do Tesouro Direto
    println("\n--- Pontos usados para curva Tesouro Direto ---")
    tesouro_points_df = DataFrame(
        "Tipo Titulo" => [b.name for b in tesouro_bonds], # CORRIGIDO: b.bond_type -> b.name
        "Data Vencimento" => [b.maturity for b in tesouro_bonds], # CORRIGIDO: Acessa o campo correto
        "PU" => [round(b.price, digits=6) for b in tesouro_bonds] # CORRIGIDO: Acessa o campo correto
    )
    pretty_table(tesouro_points_df, header=names(tesouro_points_df))
    
    # Constrói a curva do Tesouro Direto
    tesouro_curve = build_complete_yield_curve(tesouro_bonds, ref_date)
    println("Curva do Tesouro Direto construída com $(length(tesouro_curve)) pontos.")
    
    # Plota a curva do Tesouro Direto
    plot_yield_curve(tesouro_curve, ref_date, tesouro_direto_manha, output_file="yield_curve_tesouro_direto_manha.png")
    
    # Carrega dados do BACEN
    println("\n=== Carregando dados do BACEN ===")
    bacen_df = get_bacen_data(ref_date)
    bacen_bonds = prepare_bonds(bacen_df, ref_date, bacen_avg_price)
    println("Processando $(length(bacen_bonds)) títulos do BACEN...")

    # Imprime os pontos usados para a curva do BACEN
    println("\n--- Pontos usados para curva BACEN ---")
    bacen_points_df = DataFrame(
        "Tipo Titulo" => [b.name for b in bacen_bonds], # CORRIGIDO: b.bond_type -> b.name
        "Data Vencimento" => [b.maturity for b in bacen_bonds], # CORRIGIDO: Acessa o campo correto
        "PU" => [round(b.price, digits=6) for b in bacen_bonds] # CORRIGIDO: Acessa o campo correto
    )
    pretty_table(bacen_points_df, header=names(bacen_points_df))
    
    # Constrói a curva do BACEN
    bacen_curve = build_complete_yield_curve(bacen_bonds, ref_date)
    println("Curva do BACEN construída com $(length(bacen_curve)) pontos.")
    
    # Plota a curva do BACEN
    plot_yield_curve(bacen_curve, ref_date, bacen_avg_price, output_file="yield_curve_bacen_avg_price.png")
    
    # Prepara a tabela comparativa com termos específicos
    terms = collect(0.5:0.5:10.0)  # De 0.5 a 10 anos em passos de 0.5
    comparison_table = compare_curves(
        tesouro_curve,
        bacen_curve,
        ref_date,
        terms,
        curve1_name="Tesouro Direto (%)",
        curve2_name="BACEN (%)"
    )
    
    # Exibe a tabela
    println("\n=== Tabela Comparativa de Taxas (%) ===")
    pretty_table(comparison_table, header=names(comparison_table))

    # Salva a tabela comparativa em Excel, se o nome do arquivo for fornecido
    if output_excel_file !== nothing
        try
            XLSX.writetable(output_excel_file, comparison_table, sheetname="Comparacao Curvas $ref_date", overwrite=true)
            println("\nTabela comparativa salva em: $output_excel_file")
        catch e
            @error "Erro ao salvar tabela em Excel: $e"
        end
    end
    
    return tesouro_curve, bacen_curve, comparison_table
end

# Executa se chamado diretamente
if abspath(PROGRAM_FILE) == @__FILE__
    # Define a data de referência como 10/04/2025
    ref_date = "10/04/2025"
    excel_filename = "comparison_curves_$(replace(ref_date, "/" => "-")).xlsx" # Nome do arquivo Excel
    
    # Chama a função principal, passando a data e o nome do arquivo Excel
    main(ref_date=ref_date, output_excel_file=excel_filename)
end