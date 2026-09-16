#!/usr/bin/env julia
"""
copom_wirp.jl - Cortes de Selic precificados pela curva, reunião a reunião

Equivalente ao WIRP do terminal: mostra quanto de corte (ou alta) a curva
prefixada embute em cada reunião do Copom, e confronta com a mediana do Focus
para as mesmas reuniões.

Uso:
    julia --project=. analysis/copom_wirp.jl [data] [n_reunioes]

Padrão: última data com curva disponível, 8 reuniões.

A curva vem do banco histórico quando a data já está lá; caso contrário é
ajustada na hora (leva alguns segundos).
"""

using PQRateCurve
using Dates, DataFrames, CSV, Printf, Statistics, HTTP, JSON

const DB_PATH = "historical_curves.db"
const FOCUS_MEET_CACHE = joinpath("raw", "focus_selic_reunioes.csv")

REF_DATE = length(ARGS) >= 1 ? Date(ARGS[1]) : nothing
N_MEET = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 8

# ---------------------------------------------------------------------------
# Focus por reunião
# ---------------------------------------------------------------------------

"""
    fetch_focus_meetings(since) -> DataFrame

Medianas do Focus para a Selic por reunião do Copom (rótulos R1/2026 etc.).
"""
function fetch_focus_meetings(since::Date)
    base = "https://olinda.bcb.gov.br/olinda/servico/Expectativas/versao/v1/odata/ExpectativasMercadoSelic"
    rows = NamedTuple{(:release, :meeting_label, :median_pct, :n_respondents),
                      Tuple{Date,String,Float64,Int}}[]
    skip, page = 0, 1000
    while true
        url = base * "?" * HTTP.escapeuri(Dict(
            "\$filter" => "Indicador eq 'Selic' and baseCalculo eq 0 and Data ge '$(since)'",
            "\$select" => "Data,Reuniao,Mediana,numeroRespondentes",
            "\$orderby" => "Data asc",
            "\$format" => "json",
            "\$top" => string(page), "\$skip" => string(skip)))
        batch = JSON.parse(String(HTTP.get(url; retries = 3, readtimeout = 90).body))["value"]
        isempty(batch) && break
        for r in batch
            push!(rows, (release = Date(r["Data"]), meeting_label = r["Reuniao"],
                         median_pct = Float64(r["Mediana"]),
                         n_respondents = Int(r["numeroRespondentes"])))
        end
        length(batch) < page && break
        skip += page
    end
    return DataFrame(rows)
end

function load_focus_meetings(since::Date; refresh::Bool = false)
    if !refresh && isfile(FOCUS_MEET_CACHE)
        c = CSV.read(FOCUS_MEET_CACHE, DataFrame)
        !isempty(c) && maximum(c.release) >= since && return c
    end
    df = fetch_focus_meetings(since)
    mkpath(dirname(FOCUS_MEET_CACHE))
    CSV.write(FOCUS_MEET_CACHE, df)
    return df
end

"""
    meeting_label(calendar, meeting) -> String

Rótulo do Focus para uma reunião: "R<n>/<ano>", com n a ordem dentro do ano.
"""
function meeting_label(calendar::Vector{CopomMeeting}, m::CopomMeeting)
    do_ano = filter(x -> year(x.date) == year(m.date) && !x.extraordinary, calendar)
    n = findfirst(x -> x.date == m.date, do_ano)
    return "R$(n)/$(year(m.date))"
end

# ---------------------------------------------------------------------------
# Curva do dia
# ---------------------------------------------------------------------------

"""Parâmetros NSS para a data: do banco, ou ajustados na hora."""
function curve_params(ref::Date)
    if isfile(DB_PATH)
        db = init_database(DB_PATH)
        c = load_curve(db, ref)
        close(db)
        if c !== nothing && c.success == 1
            return [c.beta0, c.beta1, c.beta2, c.beta3, c.tau1, c.tau2], "banco"
        end
    end
    println("   curva não está no banco; ajustando na hora...")
    results, _ = fit_curves_for_period(ref, ref; output_csv = nothing, db_path = nothing,
                                       find_continuity = true, verbose = false)
    r = first(filter(x -> x.success, results))
    return r.params, "ajuste on-the-fly"
end

"""Última data útil com dados disponíveis."""
function latest_data_date()
    d = today()
    for _ in 1:15
        df = try
            load_bacen_data(d, d)
        catch
            DataFrame()
        end
        !isempty(df) && return d
        d -= Day(1)
    end
    error("Nenhuma data com dados nos últimos 15 dias")
end

# ---------------------------------------------------------------------------

println("=" ^ 74)
println("📉 CORTES DE SELIC PRECIFICADOS PELA CURVA  (equivalente ao WIRP)")
println("=" ^ 74)

cal = load_copom_calendar()
ref = REF_DATE === nothing ? latest_data_date() : REF_DATE

params, origem = curve_params(ref)
selic = current_selic(cal, ref)

@printf("\n📅 Curva de %s (%s)\n", ref, origem)
@printf("🎯 Selic vigente: %.2f%% a.a.\n", selic)

upcoming = next_meetings(cal, ref; n = N_MEET + 1)
path = implied_selic_path(params, ref, upcoming; current_rate = selic)

spot = metadata(path, "spot_period")
@printf("\n🔎 Diagnóstico da ponta curta: a curva implica %.2f%% para o período até a\n", spot)
@printf("   próxima vigência, contra %.2f%% de Selic vigente (%+.0f bps).\n", selic, (spot - selic) * 100)
println("   Esse gap mistura extrapolação NSS abaixo de 3 meses com o spread entre")
println("   título público e CDI — não é corte precificado.")

# Focus para as mesmas reuniões
focus = load_focus_meetings(ref - Month(2))
last_rel = maximum(focus.release)
focus_last = Dict(r.meeting_label => r.median_pct
                  for r in eachrow(filter(x -> x.release == last_rel, focus)))

println("\n" * "-" ^ 74)
@printf("%-12s %6s  %10s  %9s  %8s   %10s  %8s\n",
        "reunião", "nº", "curva %", "mov bps", "cortes", "Focus %", "dif bps")
println("-" ^ 74)

for (i, row) in enumerate(eachrow(path))
    m = upcoming[i]
    lbl = meeting_label(cal, m)
    f = get(focus_last, lbl, missing)
    if f === missing
        @printf("%-12s %6d  %10.2f  %+9.0f  %8.1f   %10s  %8s\n",
                string(row.meeting_date), row.meeting_number, row.implied_selic,
                row.move_bps, row.n_cuts_25, "-", "-")
    else
        @printf("%-12s %6d  %10.2f  %+9.0f  %8.1f   %10.2f  %+8.0f\n",
                string(row.meeting_date), row.meeting_number, row.implied_selic,
                row.move_bps, row.n_cuts_25, f, (row.implied_selic - f) * 100)
    end
end
println("-" ^ 74)

total = last(path.cumulative_bps)
@printf("\n📊 Acumulado até %s: %+.0f bps (%.1f cortes de 25 bps)\n",
        last(path.meeting_date), total, total / 25)
@printf("   Selic terminal precificada: %.2f%% a.a.\n", last(path.implied_selic))
@printf("   Focus (coleta de %s): ", last_rel)
let lbl = meeting_label(cal, upcoming[nrow(path)])
    f = get(focus_last, lbl, missing)
    f === missing ? println("sem cotação para $lbl") : @printf("%.2f%% a.a. em %s\n", f, lbl)
end

isdir("outputs") || mkpath("outputs")
CSV.write(joinpath("outputs", "copom_wirp.csv"), path)
println("\n💾 outputs/copom_wirp.csv")
