"""
test_meeting_curve.jl - Validação do bootstrap com degraus nas reuniões

Round-trip: parte de uma trajetória de Selic conhecida, gera as taxas zero
exatas que ela implica nos vencimentos observados, e verifica se o bootstrap
recupera os degraus originais.
"""

using PQRateCurve
using Dates, DataFrames, LinearAlgebra, Statistics, BusinessDays
using Test

const CALT = BusinessDays.BRSettlement()

fake_meetings(dates) = [CopomMeeting(d, 900 + i, nothing, false, true)
                        for (i, d) in enumerate(dates)]

"""Taxa zero exata até `maturity` implicada por degraus `steps` com fronteiras `bounds`."""
function exact_zero(ref::Date, bounds::Vector{Date}, steps::Vector{Float64}, maturity::Date)
    acc = 0.0
    K = length(bounds)
    for k in 1:K
        s = min(bounds[k], maturity)
        e = k < K ? min(bounds[k+1], maturity) : maturity
        e > s || continue
        acc += BusinessDays.bdayscount(CALT, s, e) / 252 * log(1 + steps[k] / 100)
    end
    du = BusinessDays.bdayscount(CALT, ref, maturity)
    return (exp(acc * 252 / du) - 1) * 100
end

@testset "Bootstrap com degraus nas reuniões" begin

    ref = Date(2026, 9, 16)
    mdates = [Date(2026, 11, 4), Date(2026, 12, 9), Date(2027, 1, 27), Date(2027, 3, 17)]
    meetings = fake_meetings(mdates)
    bounds = vcat(ref, [d + Day(1) for d in mdates])
    steps = [14.00, 13.75, 13.25, 13.25, 12.75]     # corte, corte 50, manutenção, corte 50

    @testset "round-trip: recupera degraus quando identificados" begin
        # Um vencimento logo após cada reunião: sistema exatamente determinado
        mats = [d + Day(20) for d in mdates]
        obs = [ZeroObservation(m, exact_zero(ref, bounds, steps, m), 1e6) for m in mats]

        mc = bootstrap_meeting_curve(ref, obs, meetings; current_rate = steps[1])

        @test mc.forward ≈ steps atol = 1e-6
        @test maximum(abs, mc.fit_error_bps) < 0.01
        @test all(mc.identified)

        path = implied_path(mc)
        @test path.implied_selic ≈ steps[2:end] atol = 1e-6
        @test path.move_bps ≈ [-25.0, -50.0, 0.0, -50.0] atol = 1e-4
        @test path.n_cuts_25 ≈ [-1.0, -2.0, 0.0, -2.0] atol = 1e-4
        @test path.cumulative_bps ≈ [-25.0, -75.0, -75.0, -125.0] atol = 1e-4
    end

    @testset "o primeiro degrau é a Selic conhecida, não estimada" begin
        mats = [d + Day(20) for d in mdates]
        obs = [ZeroObservation(m, exact_zero(ref, bounds, steps, m), 1e6) for m in mats]
        mc = bootstrap_meeting_curve(ref, obs, meetings; current_rate = steps[1])
        @test mc.forward[1] ≈ steps[1] atol = 1e-12

        # Mesmo com instrumentos ruidosos, o 1º degrau não se move
        ruido = [ZeroObservation(o.maturity, o.rate + 0.3, o.quantity) for o in obs]
        mc2 = bootstrap_meeting_curve(ref, ruido, meetings; current_rate = steps[1])
        @test mc2.forward[1] ≈ steps[1] atol = 1e-12
    end

    @testset "não inventa movimento onde não há instrumento" begin
        # Um único vencimento curto: só o 1º degrau estimado é disciplinado
        curto = Date(2026, 12, 20)
        obs = [ZeroObservation(curto, exact_zero(ref, bounds, steps, curto), 1e6)]

        mc = bootstrap_meeting_curve(ref, obs, meetings; current_rate = steps[1])

        # Reprecifica o instrumento que existe
        @test maximum(abs, mc.fit_error_bps) < 0.5

        # Só o segmento que CONTÉM o vencimento é individualmente identificado.
        # bounds = [16/09, 05/11, 10/12, 28/01, 18/03]; o título vence em 20/12,
        # dentro do 3º segmento [10/12, 28/01).
        @test mc.identified[1]            # 1º é a Selic fixada
        @test mc.identified[3]
        @test !mc.identified[2]
        @test !any(mc.identified[4:end])

        # Depois do último vencimento nada restringe a curva, e a solução de
        # mínimo movimento mantém o nível em vez de inventar corte
        path = implied_path(mc)
        depois = path.meeting_date .> Date(2026, 12, 20)
        @test all(abs.(path.move_bps[depois]) .< 1e-6)
    end

    @testset "curva plana devolve degraus planos" begin
        flat = 13.5
        mats = [d + Day(20) for d in mdates]
        obs = [ZeroObservation(m, flat, 1e6) for m in mats]
        mc = bootstrap_meeting_curve(ref, obs, meetings; current_rate = flat)
        @test all(isapprox.(mc.forward, flat; atol = 1e-6))
        @test all(abs.(implied_path(mc).move_bps) .< 1e-4)
    end

    @testset "a curva reconstruída reprecifica os instrumentos" begin
        mats = [d + Day(20) for d in mdates]
        obs = [ZeroObservation(m, exact_zero(ref, bounds, steps, m), 1e6) for m in mats]
        mc = bootstrap_meeting_curve(ref, obs, meetings; current_rate = steps[1])

        zr = zero_rate_curve(mc)
        for o in obs
            t = BusinessDays.bdayscount(CALT, ref, o.maturity) / 252
            implied = (exp(zr(t)) - 1) * 100
            @test implied ≈ o.rate atol = 1e-6
        end

        # E encaixa na mecânica genérica de forwards inter-reunião
        p2 = implied_selic_path(zr, ref, meetings; current_rate = steps[1])
        @test p2.implied_selic ≈ implied_path(mc).implied_selic[1:nrow(p2)] atol = 1e-6
    end

    @testset "LTN: taxa zero sai do preço, sem ajuste" begin
        mat = Date(2027, 1, 1)
        du = BusinessDays.bdayscount(CALT, ref, mat)
        taxa = 13.4
        pu = 1000 / (1 + taxa / 100)^(du / 252)

        df = DataFrame(date = [ref], bond_code = ["LTN"], maturity_date = [mat],
                       avg_price = [pu], quantity_traded = [5000.0])
        obs = ltn_zero_rates(df, ref)
        @test length(obs) == 1
        @test obs[1].rate ≈ taxa atol = 1e-9
        @test obs[1].maturity == mat

        # NTN-F não é zero-cupom e tem de ser ignorada aqui
        df2 = DataFrame(date = [ref], bond_code = ["NTN-F"], maturity_date = [mat],
                        avg_price = [1000.0], quantity_traded = [10.0])
        @test isempty(ltn_zero_rates(df2, ref))
    end

    @testset "validação de entradas" begin
        obs = [ZeroObservation(Date(2027, 1, 1), 13.0, 1.0)]
        @test_throws ArgumentError bootstrap_meeting_curve(ref, ZeroObservation[], meetings; current_rate = 14.0)
        @test_throws ArgumentError bootstrap_meeting_curve(ref, obs, CopomMeeting[]; current_rate = 14.0)
        passadas = fake_meetings([Date(2026, 1, 28)])
        @test_throws ArgumentError bootstrap_meeting_curve(ref, obs, passadas; current_rate = 14.0)
        # Instrumento já vencido
        vencido = [ZeroObservation(Date(2026, 1, 1), 13.0, 1.0)]
        @test_throws ArgumentError bootstrap_meeting_curve(ref, vencido, meetings; current_rate = 14.0)
    end
end
