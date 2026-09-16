"""
test_b3_curves.jl - Leitura das curvas de referência da B3

Usa os arquivos já baixados em raw/b3_ts (imutáveis uma vez publicados), então
não depende de rede. Se o cache estiver vazio, os testes que precisam dele são
pulados com aviso, em vez de falharem por indisponibilidade da B3.
"""

using PQRateCurve
using Dates, DataFrames, BusinessDays, Statistics
using Test

const CACHE = joinpath("raw", "b3_ts")
const CALB = BusinessDays.BRSettlement()

"""Uma data com arquivo em cache, ou `nothing`."""
function cached_date()
    isdir(CACHE) || return nothing
    files = filter(readdir(CACHE)) do f
        startswith(f, "TS") && endswith(f, ".zip") && filesize(joinpath(CACHE, f)) > 1000
    end
    isempty(files) && return nothing
    m = match(r"TS(\d{2})(\d{2})(\d{2})\.zip", last(sort(files)))
    m === nothing ? nothing :
        Date(2000 + parse(Int, m[1]), parse(Int, m[2]), parse(Int, m[3]))
end

@testset "Curvas de referência da B3" begin

    ref = cached_date()

    if ref === nothing
        @warn "raw/b3_ts vazio — testes de leitura pulados"
    else
        @testset "leitura da curva PRE" begin
            df = fetch_b3_curve(ref)

            @test nrow(df) > 200                       # a PRE tem ~272 vértices
            @test names(df) == ["maturity", "calendar_days", "business_days", "rate"]
            @test issorted(df.business_days)
            @test all(df.business_days .> 0)
            @test all(df.calendar_days .>= df.business_days)

            # Taxas em nível plausível para o Brasil, em % a.a.
            @test all(0 .< df.rate .< 60)

            # O vencimento tem de bater com os dias corridos
            @test df.maturity[1] == ref + Day(df.calendar_days[1])

            # Densidade na ponta curta é o motivo de usar essa fonte
            @test count(<=(504), df.business_days) > 80
        end

        @testset "curva inexistente é recusada" begin
            @test_throws ArgumentError fetch_b3_curve(ref; curve = "XXX")
        end

        @testset "interpolação flat-forward" begin
            df = fetch_b3_curve(ref)

            # Nos vértices, devolve o valor exato
            for i in (1, 10, 60, nrow(df))
                @test interpolate_flat_forward(df, df.business_days[i]) ≈ df.rate[i] atol = 1e-9
            end

            # Entre vértices, fica entre os vizinhos
            for i in (5, 40, 100)
                lo, hi = minmax(df.rate[i], df.rate[i+1])
                mid = interpolate_flat_forward(df, (df.business_days[i] + df.business_days[i+1]) / 2)
                @test lo - 1e-9 <= mid <= hi + 1e-9
            end

            # Fora do intervalo, estende as pontas
            @test interpolate_flat_forward(df, 0) ≈ df.rate[1]
            @test interpolate_flat_forward(df, -5) ≈ df.rate[1]
            @test interpolate_flat_forward(df, 10^6) ≈ df.rate[end]

            # Forward constante entre dois vértices: o fator de capitalização
            # interpolado tem de ser log-linear em dias úteis
            i = 50
            d1, d2 = df.business_days[i], df.business_days[i+1]
            if d2 > d1 + 1
                dm = (d1 + d2) / 2
                f(du, r) = log(1 + r / 100) * du / 252
                @test f(dm, interpolate_flat_forward(df, dm)) ≈
                      (f(d1, df.rate[i]) + f(d2, df.rate[i+1])) / 2 atol = 1e-12
            end
        end
    end

    @testset "catálogo de curvas" begin
        @test haskey(B3_REFERENCE_CURVES, "PRE")
        @test all(length.(keys(B3_REFERENCE_CURVES)) .<= 3)
    end
end
