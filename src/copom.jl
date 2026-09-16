"""
copom.jl - Cortes de Selic precificados pela curva (equivalente ao WIRP)

Entre duas reuniões do Copom a Selic é constante, então a taxa forward de cada
período inter-reunião *é* a Selic implícita para aquele período. A diferença
entre períodos consecutivos é o movimento que o mercado precifica para a
reunião que os separa, e dividido por 25 bps dá o número de cortes (ou altas).

Convenção brasileira em tudo: taxas efetivas anuais capitalizadas em 252 dias
úteis, com fator de desconto (1 + i)^(-du/252). A curva NSS do projeto trabalha
em taxa contínua sobre a mesma base 252, então a ponte é i = exp(r) - 1.

Detalhe de calendário que muda o resultado: a decisão sai no segundo dia da
reunião, mas a nova taxa só vigora no dia seguinte (o BC divulga
DataInicioVigencia = data da reunião + 1). São as datas de *vigência*, não as
de reunião, que delimitam os períodos forward.

Referência de agenda: `raw/copom_calendar.csv` (ver `load_copom_calendar`).
"""

# ============================================================================
# Agenda do Copom
# ============================================================================

"""
    CopomMeeting

Uma reunião do Copom. `selic_target` é a meta decidida (apenas para reuniões
passadas); `scheduled` marca reuniões futuras, ainda sem decisão.
"""
struct CopomMeeting
    date::Date                              # data da decisão (2º dia da reunião)
    number::Int
    selic_target::Union{Float64,Nothing}    # % a.a., meta definida na reunião
    extraordinary::Bool
    scheduled::Bool
end

"""
    effective_date(m::CopomMeeting) -> Date

Data em que a taxa decidida passa a vigorar: o dia seguinte à decisão.
"""
effective_date(m::CopomMeeting) = m.date + Day(1)

"""
    load_copom_calendar(path="raw/copom_calendar.csv") -> Vector{CopomMeeting}

Carrega a agenda do Copom, ordenada por data.

O arquivo reúne o histórico completo (fonte: endpoint `historicotaxasjuros` do
BC, que inclui reuniões que mantiveram a taxa — a série SGS 432 não inclui) e
as reuniões já agendadas, vindas dos comunicados anuais de calendário.

A agenda futura **não é derivável**: toda reunião cai numa quarta-feira e todo
intervalo é múltiplo de 7 dias, mas qual quarta é decisão de agenda (1ª a 5ª do
mês, intervalos de 35 a 56 dias). O arquivo precisa ser estendido todo junho,
quando o BC publica o calendário do ano seguinte.
"""
function load_copom_calendar(path::String = joinpath("raw", "copom_calendar.csv"))
    isfile(path) || throw(ArgumentError("Calendário do Copom não encontrado: $path"))
    df = CSV.read(path, DataFrame)

    meetings = CopomMeeting[]
    for row in eachrow(df)
        target = ismissing(row.selic_target_pct) ? nothing : Float64(row.selic_target_pct)
        push!(meetings, CopomMeeting(Date(row.meeting_date), Int(row.meeting_number),
                                     target, Bool(row.extraordinary), Bool(row.scheduled)))
    end
    sort!(meetings, by = m -> m.date)
    return meetings
end

"""
    current_selic(meetings, ref_date) -> Float64

Meta Selic vigente em `ref_date`, isto é, a decidida na última reunião cuja taxa
já entrou em vigor.
"""
function current_selic(meetings::Vector{CopomMeeting}, ref_date::Date)
    idx = findlast(m -> effective_date(m) <= ref_date && m.selic_target !== nothing, meetings)
    idx === nothing && throw(ArgumentError(
        "Nenhuma meta Selic vigente em $ref_date no calendário carregado"))
    return meetings[idx].selic_target
end

"""
    next_meetings(meetings, ref_date; n=8) -> Vector{CopomMeeting}

As `n` próximas reuniões cuja taxa ainda não vigorava em `ref_date`.
"""
function next_meetings(meetings::Vector{CopomMeeting}, ref_date::Date; n::Int = 8)
    upcoming = filter(m -> effective_date(m) > ref_date, meetings)
    return upcoming[1:min(n, length(upcoming))]
end

# ============================================================================
# Extração da trajetória implícita
# ============================================================================

"""
    implied_selic_path(zero_rate, ref_date, meetings; current_rate=nothing) -> DataFrame

Trajetória de Selic que a curva precifica, reunião a reunião.

`zero_rate(t)` devolve a taxa zero **contínua anualizada** para o prazo `t` em
anos na base 252 (é a assinatura de `nss_rate(t, params)`). `meetings` são as
próximas reuniões, em ordem.

Para cada período entre vigências consecutivas calcula a forward

    f = [r(t₂)·t₂ - r(t₁)·t₁] / (t₂ - t₁)

e a converte para taxa efetiva anual, `i = exp(f) - 1`, que é a Selic implícita
daquele período.

Colunas devolvidas:
- `meeting_date`, `meeting_number`
- `implied_selic`: Selic precificada para o período que se inicia nessa reunião (% a.a.)
- `move_bps`: variação em relação ao período anterior
- `cumulative_bps`: variação acumulada desde a Selic corrente
- `n_cuts_25`: `move_bps` em múltiplos de 25 bps (negativo = corte)

`current_rate` (% a.a.) é a Selic vigente. Quando informada, a primeira linha
mede o movimento contra ela; a taxa implícita do período que vai de `ref_date`
até a primeira vigência é devolvida como atributo `:spot_period` do DataFrame e
deve bater com ela — a diferença é um bom diagnóstico da ponta curta da curva.
"""
function implied_selic_path(zero_rate, ref_date::Date, meetings::Vector{CopomMeeting};
                            current_rate::Union{Float64,Nothing} = nothing)
    isempty(meetings) && throw(ArgumentError("Nenhuma reunião fornecida"))
    all(m -> effective_date(m) > ref_date, meetings) ||
        throw(ArgumentError("Há reuniões cuja vigência não é posterior a $ref_date"))
    issorted(meetings, by = m -> m.date) ||
        throw(ArgumentError("As reuniões precisam estar em ordem cronológica"))

    # Fronteiras dos períodos: hoje e cada data de vigência
    bounds = vcat(ref_date, [effective_date(m) for m in meetings])
    ts = [yearfrac(ref_date, d) for d in bounds]          # anos base 252
    ts[1] = 0.0

    any(diff(ts) .<= 0) && throw(ArgumentError(
        "Datas de vigência sem dias úteis entre elas — calendário inconsistente"))

    # log do fator de desconto acumulado até cada fronteira
    logdf = [t == 0 ? 0.0 : -zero_rate(t) * t for t in ts]

    return _assemble_path(meetings, ts, logdf, ref_date, current_rate)
end

"""
    _assemble_path(meetings, ts, logdf, ref_date, current_rate) -> DataFrame

Monta o DataFrame de saída a partir dos fatores de desconto nas fronteiras.
"""
function _assemble_path(meetings, ts, logdf, ref_date, current_rate)
    n = length(meetings)

    # Períodos: [ref, e₁), [e₁, e₂), ..., [e_{n-1}, e_n)
    # O primeiro é o período corrente (Selic já conhecida); os demais começam
    # em cada reunião. A última reunião não tem período à frente delimitado,
    # então o movimento dela é medido até a reunião seguinte, que não existe —
    # por isso só há informação para n-1 reuniões mais o período corrente.
    period_rate = Vector{Float64}(undef, n)
    for i in 1:n
        f = -(logdf[i+1] - logdf[i]) / (ts[i+1] - ts[i])
        period_rate[i] = (exp(f) - 1) * 100
    end

    spot_period = period_rate[1]      # de hoje até a 1ª vigência: a Selic corrente
    base = current_rate === nothing ? spot_period : current_rate

    rows = DataFrame(meeting_date = Date[], meeting_number = Int[],
                     implied_selic = Float64[], move_bps = Float64[],
                     cumulative_bps = Float64[], n_cuts_25 = Float64[])

    prev = base
    for i in 1:(n-1)
        m = meetings[i]
        rate = period_rate[i+1]       # taxa que vigora após a reunião i
        move = (rate - prev) * 100
        push!(rows, (m.date, m.number, rate, move, (rate - base) * 100, move / 25))
        prev = rate
    end

    metadata!(rows, "spot_period", spot_period, style = :note)
    metadata!(rows, "current_rate", base, style = :note)
    metadata!(rows, "ref_date", ref_date, style = :note)
    return rows
end

"""
    implied_selic_path(params::Vector{Float64}, ref_date, meetings; kwargs...) -> DataFrame

Versão que parte dos parâmetros NSS de uma curva ajustada.
"""
implied_selic_path(params::Vector{Float64}, ref_date::Date,
                   meetings::Vector{CopomMeeting}; kwargs...) =
    implied_selic_path(t -> nss_rate(t, params), ref_date, meetings; kwargs...)

"""
    implied_selic_path(db::SQLite.DB, ref_date; n=8, calendar=nothing) -> DataFrame

Versão que busca a curva do dia no banco e a agenda em disco.
"""
function implied_selic_path(db::SQLite.DB, ref_date::Date;
                            n::Int = 8,
                            calendar::Union{Vector{CopomMeeting},Nothing} = nothing)
    curve = load_curve(db, ref_date)
    curve === nothing && throw(ArgumentError("Sem curva no banco para $ref_date"))
    curve.success == 1 || throw(ArgumentError("A curva de $ref_date falhou no ajuste"))

    cal = calendar === nothing ? load_copom_calendar() : calendar
    params = [curve.beta0, curve.beta1, curve.beta2, curve.beta3, curve.tau1, curve.tau2]

    # n+1 reuniões: a última só serve para fechar o período da penúltima
    upcoming = next_meetings(cal, ref_date; n = n + 1)
    length(upcoming) >= 2 || throw(ArgumentError(
        "Agenda insuficiente após $ref_date: só $(length(upcoming)) reunião(ões) cadastrada(s). " *
        "Estenda raw/copom_calendar.csv com o calendário publicado pelo BC."))

    return implied_selic_path(params, ref_date, upcoming;
                              current_rate = current_selic(cal, ref_date))
end
