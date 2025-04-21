using HTTP
using CSV
using DataFrames
using Dates
using XLSX
using Gumbo
using Cascadia

function load_treasury_data()
    # URL do dataset
    url = "https://www.tesourotransparente.gov.br/ckan/dataset/df56aa42-484a-4a59-8184-7676580c81e3/resource/796d2059-14e9-44e3-80c9-2d9e30b405c1/download/PrecoTaxaTesouroDireto.csv"
    
    try
        # Baixar o conteúdo do CSV
        println("Baixando dados do Tesouro Direto de $url")
        response = HTTP.get(url; retries=3, readtimeout=30)
        csv_data = String(response.body)
        
        # Carregar CSV diretamente
        df = CSV.read(IOBuffer(csv_data), DataFrame; delim=';', decimal=',')
        println("Dataset carregado com $(nrow(df)) registros")
        return df
    catch e
        println("Erro ao carregar dados do Tesouro Direto: $e")
        # Retorna um DataFrame vazio em caso de erro
        return DataFrame() 
    end
end

function filter_treasury_bonds(df, base_date; type=nothing, coupon=nothing)
    # Filtra os registros pela data base
    filtered_df = df[df[!, "Data Base"] .== base_date, :]
    
    # Filtra por tipo se especificado
    if !isnothing(type)
        if type == "PRE"
            filtered_df = filtered_df[occursin.("Prefixado", filtered_df[!, "Tipo Titulo"]), :]
        elseif type == "IPCA"
            filtered_df = filtered_df[occursin.("IPCA", filtered_df[!, "Tipo Titulo"]) .| 
                                    occursin.("IGPM", filtered_df[!, "Tipo Titulo"]) .|
                                    occursin.("Renda", filtered_df[!, "Tipo Titulo"]) .|
                                    occursin.("Educa", filtered_df[!, "Tipo Titulo"]), :]
        elseif type == "POS"
            filtered_df = filtered_df[occursin.("Selic", filtered_df[!, "Tipo Titulo"]), :]
        end
    end
    
    # Filtra por cupom se especificado
    if !isnothing(coupon)
        if coupon
            filtered_df = filtered_df[occursin.("Juros Semestrais", filtered_df[!, "Tipo Titulo"]), :]
        else
            filtered_df = filtered_df[.!occursin.("Juros Semestrais", filtered_df[!, "Tipo Titulo"]), :]
        end
    end
    
    return filtered_df
end

function load_ntnb_nominal_values()
    try
        # URL da página que contém o link de download
        url = "https://sisweb.tesouro.gov.br/apex/f?p=2501:9::::9:P9_ID_PUBLICACAO:28715"
        
        println("Tentando acessar $url para buscar valores nominais NTN-B")
        # Baixar e parsear a página HTML
        response = HTTP.get(url, redirect=true, retries=2, readtimeout=15)
        html = parsehtml(String(response.body))
        
        # Encontrar o frame de download
        frames = eachmatch(Selector("frame"), html.root)
        
        if isempty(frames)
            println("Frame de download não encontrado")
            return DataFrame(Data = Date[], Valor_Nominal = Float64[])
        end
        
        # Pega o primeiro frame encontrado
        frame = frames[1]
        
        # Extrair URL do frame
        xlsx_link = getattr(frame, "src")
        
        if isempty(xlsx_link)
            println("URL de download não encontrado no frame")
            return DataFrame(Data = Date[], Valor_Nominal = Float64[])
        end
        
        println("Baixando arquivo de: ", xlsx_link)
        
        # Verificar tipo de conteúdo
        response = HTTP.head(xlsx_link, retries=2, readtimeout=15)
        content_type = HTTP.header(response, "Content-Type")
        
        if !occursin("excel", lowercase(content_type)) && !occursin("spreadsheet", lowercase(content_type))
            println("Tipo de conteúdo inválido: $content_type")
            return DataFrame(Data = Date[], Valor_Nominal = Float64[])
        end
        
        # Baixar o arquivo XLSX
        response = HTTP.get(xlsx_link, redirect=true, retries=2, readtimeout=15)
        
        # Salvar arquivo temporariamente
        temp_file = "temp.xlsx"
        open(temp_file, "w") do io
            write(io, Vector{UInt8}(response.body))
        end
        
        # Ler arquivo Excel
        xf = XLSX.readxlsx(temp_file)
        sheet = xf[1]  # Pega a primeira planilha
        
        # Ler dados a partir da linha 11 até o final da planilha
        dim = string(XLSX.get_dimension(sheet))  # Converte CellRange para string
        last_row = parse(Int, split(dim, ":")[2][2:end])
        df = sheet["A11:B$last_row"]
        df = DataFrame(df, [:Data, :Valor_Nominal])
        
        # Remove linhas vazias
        filter!(row -> !all(ismissing, row), df)
        
        # Remove arquivo temporário
        rm(temp_file)
        
        return df
    catch e
        println("Erro ao carregar valores nominais das NTN-Bs: $e")
        # Retorna um DataFrame vazio em caso de erro
        return DataFrame(Data = Date[], Valor_Nominal = Float64[])
    end
end

# Este bloco só executa se o arquivo for executado diretamente (não quando incluído)
if abspath(PROGRAM_FILE) == @__FILE__
    println("Carregando dados do Tesouro Direto...")
    df_treasury = load_treasury_data()
    println("Dataset carregado com ", nrow(df_treasury), " registros")
    
    println("\nCarregando valores nominais das NTN-Bs...")
    df_ntnb = load_ntnb_nominal_values()
    println("Dataset NTN-B carregado com ", nrow(df_ntnb), " registros")
    
    # Teste da função filter_treasury_bonds
    println("\nTestando filter_treasury_bonds...")
    bonds_by_date = filter_treasury_bonds(df_treasury, "11/04/2025"; type="PRE")
    println(bonds_by_date)
end

