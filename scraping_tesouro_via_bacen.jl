using HTTP
using ZipFile
using CSV
using DataFrames
using Dates

"""
    inspect_csv_from_zip(zip_url::String, num_lines::Int=10)

Baixa um arquivo ZIP, encontra o primeiro CSV dentro dele e imprime 
as primeiras `num_lines` linhas para inspeção.
"""
function inspect_csv_from_zip(zip_url::String, num_lines::Int=10)
    println("Baixando: $zip_url")
    try
        response = HTTP.get(zip_url; retries=2, readtimeout=60)
        zip_content = response.body
        zip_reader = ZipFile.Reader(IOBuffer(zip_content))

        csv_file = nothing
        for file in zip_reader.files
            if endswith(lowercase(file.name), ".csv")
                csv_file = file
                break
            end
        end

        if isnothing(csv_file)
            println("Nenhum arquivo CSV encontrado no ZIP.")
            close(zip_reader)
            return
        end

        println("Inspecionando as primeiras $num_lines linhas de: $(csv_file.name)")
        
        # Lê e imprime as primeiras linhas como texto bruto
        io = IOBuffer()
        write(io, read(csv_file))
        seekstart(io)
        line_count = 0
        for line in eachline(io)
            line_count += 1
            println(line)
            if line_count >= num_lines
                break
            end
        end
        
        close(zip_reader)

    catch e
        println("Erro ao baixar ou inspecionar o arquivo: $e")
    end
end


"""
    load_bacen_data(start_date::Date, end_date::Date)

Baixa, extrai e lê os dados de negociação de títulos públicos do BACEN 
para o intervalo de datas especificado.
"""
function load_bacen_data(start_date::Date, end_date::Date)
    base_url = "https://www4.bcb.gov.br/pom/demab/negociacoes/download/NegT"
    all_data = DataFrame() # DataFrame para acumular os dados de todos os meses

    current_date = Date(year(start_date), month(start_date), 1)

    while current_date <= end_date
        dfmt = DateFormat("yyyymm") # Usa 'mm' minúsculo para mês
        year_month = Dates.format(current_date, dfmt)
        zip_url = base_url * year_month * ".zip"
        
        println("Processando: $zip_url")
        
        try
            # Baixa o conteúdo do ZIP
            response = HTTP.get(zip_url; retries=2, readtimeout=60) # Aumenta timeout
            zip_content = response.body

            # Abre o ZIP em memória
            zip_reader = ZipFile.Reader(IOBuffer(zip_content))

            # Encontra o arquivo CSV dentro do ZIP (assume que há apenas um)
            csv_file = nothing
            for file in zip_reader.files
                if endswith(lowercase(file.name), ".csv")
                    csv_file = file
                    break
                end
            end

            if isnothing(csv_file)
                println("Nenhum arquivo CSV encontrado no ZIP para $year_month")
                continue # Pula para o próximo mês
            end

            println("Lendo CSV: $(csv_file.name)")
            # Lê o CSV diretamente do ZIP, especificando tipos e formatos
            temp_df = CSV.read(
                csv_file, 
                DataFrame; 
                delim=';', 
                decimal=',', 
                dateformat="d/m/y", 
                missingstring="", # Corrigido de missingstrings para missingstring
                types=Dict(
                    "DATA MOV" => Date, 
                    "VENCIMENTO" => Date, 
                    "PU MED" => Float64
                )
            )
            
            # Adiciona os dados lidos ao DataFrame principal
            append!(all_data, temp_df)
            
            # Fecha o leitor do ZIP
            close(zip_reader)

        catch e
            println("Erro ao processar o arquivo para $year_month: $e")
            # Continua para o próximo mês mesmo se um falhar
        end

        # Avança para o próximo mês
        current_date = current_date + Month(1)
    end

    # Verifica se algum dado foi carregado
    if isempty(all_data)
        println("Nenhum dado foi baixado ou lido no período especificado.")
        return DataFrame() # Retorna DataFrame vazio
    end

    println("Processando $(nrow(all_data)) registros baixados...")

    # --- Processamento e Limpeza ---

    # 1. Renomear colunas (agora VENCIMENTO já é lido como Date)
    rename_map = Dict(
        "DATA MOV" => "date",
        "SIGLA" => "bond_code",
        "VENCIMENTO" => "maturity_date", # Renomeia diretamente
        "PU MED" => "avg_price"
    )
    # Renomeia apenas as colunas que existem no DataFrame lido
    rename!(all_data, Dict(Symbol(k) => Symbol(v) for (k, v) in rename_map if hasproperty(all_data, Symbol(k))))

    # Assert: Verificar se as colunas essenciais existem após renomear
    essential_cols_after_rename = [:date, :bond_code, :maturity_date, :avg_price]
    for col in essential_cols_after_rename
        if !hasproperty(all_data, col)
             error("Coluna essencial '$col' não encontrada após renomear. Colunas disponíveis: $(names(all_data))")
        end
    end

    # 2. Selecionar apenas as colunas de interesse
    select!(all_data, essential_cols_after_rename)

    # 3. Converter tipos e tratar erros - Simplificado
    #    As datas e PU MED já devem ter sido lidos com os tipos corretos
    #    Apenas verificamos e removemos missings

    println("Verificando tipos após leitura e renomeação:")
    println("  Tipo de 'date': $(eltype(all_data.date))")
    println("  Tipo de 'maturity_date': $(eltype(all_data.maturity_date))")
    println("  Tipo de 'avg_price': $(eltype(all_data.avg_price))")

    #    Remove quaisquer linhas onde dados essenciais estão faltando (missing)
    #    Isso inclui linhas onde a leitura/conversão inicial falhou
    dropmissing!(all_data, [:date, :bond_code, :maturity_date, :avg_price])

    # 4. Filtrar pelo intervalo de datas exato [start_date, end_date]
    #    Certifica que a coluna date é do tipo Date antes de filtrar
    if eltype(all_data.date) <: Date
        filter!(row -> start_date <= row.date <= end_date, all_data)
    else
        println("A coluna 'date' não é do tipo Date após as conversões. Não é possível filtrar por data.")
    end

    # 5. Selecionar e reordenar colunas finais (já feito na etapa 2, mas garantindo)
    final_cols = [:date, :bond_code, :maturity_date, :avg_price]
    select!(all_data, intersect(final_cols, propertynames(all_data)))

    println("Processamento concluído. $(nrow(all_data)) registros válidos no período.")

    return all_data
end

# Bloco para teste (será executado apenas se o script for chamado diretamente)
if abspath(PROGRAM_FILE) == @__FILE__
    # --- Inspeção Primeiro (Opcional) ---
    # test_zip_url = "https://www4.bcb.gov.br/pom/demab/negociacoes/download/NegT201001.zip"
    # println("--- Iniciando Inspeção do CSV ---")
    # inspect_csv_from_zip(test_zip_url, 15) # Imprime as 15 primeiras linhas
    # println("--- Fim da Inspeção do CSV ---")
    
    # --- Teste de Carga Completo ---
    println("\n--- Iniciando Teste de Carga Completo ---")
    test_start_date = Date(2010, 1, 1)
    test_end_date = Date(2010, 3, 31) 
    println("Período de teste: $test_start_date a $test_end_date")
    df_bacen = load_bacen_data(test_start_date, test_end_date)
    if isempty(df_bacen)
        println("Nenhum dado válido encontrado para o período de teste.")
    else
        println("Dados reais carregados e processados:")
        println(first(df_bacen, 5))
    end
    println("--- Fim do Teste de Carga ---")
end
