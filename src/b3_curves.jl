"""
b3_curves.jl - Curvas de referência da B3 (arquivo TaxaSwap)

A B3 publica diariamente, em arquivo público e sem autenticação, mais de cem
curvas de referência — entre elas a `PRE` (DI × pré), que é a curva de juros
do CDI. São 272 vértices por data, 126 deles abaixo de 24 meses: densidade
muito superior à dos títulos prefixados negociados, cuja LTN mais curta vence
trimestralmente.

Distribuição: `b3.com.br/pesquisapregao/download?filelist=TS{aammdd}.ex_`.
O arquivo é um ZIP contendo um auto-extraível PKSFX, que por sua vez contém
`TaxaSwap.txt`, de largura fixa.

Nota sobre a fonte: o portal novo da B3 (arquivos.b3.com.br/bdi) exige login e
guarda apenas 21 dias. Esta rota legada continua aberta e serve histórico longo
— verificado de 2006 até hoje. Mapeada a partir do pacote R `rb3` (rOpenSci).
"""

"""
    B3_REFERENCE_CURVES

Alguns códigos de curva úteis do arquivo TaxaSwap.
"""
const B3_REFERENCE_CURVES = Dict(
    "PRE" => "DI x pré (curva de juros do CDI)",
    "DIC" => "DI x IPCA",
    "DIM" => "DI x IGP-M",
    "DOL" => "DI x dólar (cupom cambial sujo)",
    "DCP" => "Cupom limpo",
    "TIC" => "NTN-B",
    "SLP" => "Selic x pré",
    "TR"  => "DI x TR",
)

const B3_TS_URL = "https://www.b3.com.br/pesquisapregao/download?filelist="

"""
    fetch_b3_curve(date; curve="PRE", cache_dir="raw/b3_ts") -> DataFrame

Baixa e interpreta uma curva de referência da B3 para a data.

Retorna DataFrame com `maturity` (data), `calendar_days`, `business_days` e
`rate` (% a.a., base 252) — a convenção nativa do arquivo, a mesma de
`ZeroObservation`.

O arquivo bruto é guardado em `cache_dir`, porque é imutável: uma vez publicado
para uma data, não muda.
"""
function fetch_b3_curve(date::Date; curve::String = "PRE",
                        cache_dir::String = joinpath("raw", "b3_ts"))
    mkpath(cache_dir)
    fname = "TS" * Dates.format(date, "yymmdd") * ".zip"
    path = joinpath(cache_dir, fname)

    if !isfile(path) || filesize(path) < 1000
        url = B3_TS_URL * "TS" * Dates.format(date, "yymmdd") * ".ex_"
        resp = HTTP.get(url; retries = 3, readtimeout = 120)
        # Datas sem pregão (ou no futuro) devolvem uma página de erro curta com
        # status 200. Gravá-la envenena o cache: a leitura seguinte encontra o
        # arquivo, não rebaixa, e falha para sempre.
        length(resp.body) < 1000 && throw(ArgumentError(
            "Sem arquivo de curvas da B3 para $date (dia sem pregão ou data futura)"))
        write(path, resp.body)
    end

    txt = _read_taxaswap(path)
    return _parse_curve(txt, date, curve)
end

"""
    _read_taxaswap(zip_path) -> String

Extrai `TaxaSwap.txt` do ZIP externo e do auto-extraível interno.
"""
function _read_taxaswap(zip_path::String)
    outer = ZipFile.Reader(zip_path)
    try
        inner_bytes = read(outer.files[1])
        inner = ZipFile.Reader(IOBuffer(inner_bytes))
        try
            idx = findfirst(f -> occursin("TaxaSwap", f.name), inner.files)
            idx === nothing && throw(ArgumentError("TaxaSwap.txt não encontrado em $zip_path"))
            return String(read(inner.files[idx]))
        finally
            close(inner)
        end
    finally
        close(outer)
    end
end

# Cauda de cada linha: dias corridos, dias úteis, taxa (7 casas), flag, prazo
const _TS_LINE = r"(\d{5})(\d{5})([+-]\d{14})(\w)(\d{5})\s*$"

function _parse_curve(txt::AbstractString, date::Date, curve::String)
    rows = NamedTuple{(:maturity, :calendar_days, :business_days, :rate),
                      Tuple{Date,Int,Int,Float64}}[]

    for line in split(txt, '\n')
        length(line) < 45 && continue
        line[22:24] == curve || continue
        m = match(_TS_LINE, line)
        m === nothing && continue

        cur = parse(Int, m.captures[1])
        biz = parse(Int, m.captures[2])
        rate = parse(Int, m.captures[3]) / 1e7
        (cur > 0 && biz > 0) || continue
        push!(rows, (maturity = date + Day(cur), calendar_days = cur,
                     business_days = biz, rate = rate))
    end

    isempty(rows) && throw(ArgumentError(
        "Curva '$curve' não encontrada em $date. Códigos conhecidos: " *
        join(sort(collect(keys(B3_REFERENCE_CURVES))), ", ")))

    df = DataFrame(rows)
    sort!(df, :business_days)
    return df
end

"""
    b3_curve_observations(df; max_maturity=nothing) -> Vector{ZeroObservation}

Converte a curva da B3 em observações para o bootstrap. Todos os vértices têm o
mesmo peso: são pontos de uma curva já construída pela bolsa, não negócios
individuais com liquidez distinta.
"""
function b3_curve_observations(df::DataFrame; max_maturity::Union{Date,Nothing} = nothing)
    obs = ZeroObservation[]
    for row in eachrow(df)
        max_maturity !== nothing && row.maturity > max_maturity && continue
        push!(obs, ZeroObservation(row.maturity, row.rate, 1.0))
    end
    return obs
end

# ============================================================================
# Interpolação
# ============================================================================

"""
    interpolate_flat_forward(df, business_days) -> Float64

Taxa efetiva anual no prazo pedido, interpolada da curva da B3 pelo método
flat-forward — padrão do mercado brasileiro. Interpola linearmente o logaritmo
do fator de capitalização em dias úteis, o que equivale a supor forward
constante entre vértices.

Fora do intervalo da curva, estende o último forward disponível.
"""
function interpolate_flat_forward(df::DataFrame, business_days::Real)
    business_days <= 0 && return df.rate[1]
    du = df.business_days
    business_days <= du[1] && return df.rate[1]
    business_days >= du[end] && return df.rate[end]

    i = searchsortedlast(du, business_days)
    du[i] == business_days && return df.rate[i]

    # log do fator acumulado em cada vértice
    f1 = log(1 + df.rate[i] / 100) * du[i] / 252
    f2 = log(1 + df.rate[i+1] / 100) * du[i+1] / 252
    w = (business_days - du[i]) / (du[i+1] - du[i])
    f = f1 + w * (f2 - f1)
    return (exp(f * 252 / business_days) - 1) * 100
end
