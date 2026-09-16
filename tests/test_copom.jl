"""
test_copom.jl - Validação dos cortes de Selic precificados pela curva

Estratégia: partir de uma trajetória de Selic CONHECIDA, construir
analiticamente a curva zero que ela implica, e verificar se o extrator devolve
exatamente os degraus originais. Se a mecânica de forwards, contagem de dias
úteis e conversão de convenção estiver certa, o round-trip é exato.
"""

using PQRateCurve
using Dates, DataFrames, Statistics, BusinessDays
using Test

const CAL = BusinessDays.BRSettlement()

"""
Constrói a função de taxa zero contínua (base 252) implicada por uma Selic que
vale `rates[i]` (% a.a. efetiva) entre `bounds[i]` e `bounds[i+1]`.
"""
function step_curve(ref_date::Date, bounds::Vector{Date}, rates::Vector{Float64})
    @assert length(rates) == length(bounds)   # última taxa vale daí em diante
    function zero_rate(t)
        t <= 0 && return log(1 + rates[1] / 100)
        target_du = t * 252
        acc_log, du_done, i = 0.0, 0.0, 1
        while du_done < target_du - 1e-9
            seg = if i < length(bounds)
                Float64(BusinessDays.bdayscount(CAL, bounds[i], bounds[i+1]))
            else
                Inf
            end
            take = min(seg, target_du - du_done)
            acc_log += take / 252 * log(1 + rates[i] / 100)
            du_done += take
            i += 1
        end
        return acc_log / t                     # taxa contínua média = -log(DF)/t
    end
    return zero_rate
end

"""Monta reuniões fictícias nas datas dadas."""
fake_meetings(dates) = [CopomMeeting(d, 900 + i, nothing, false, true)
                        for (i, d) in enumerate(dates)]

@testset "Copom / cortes implícitos" begin

    @testset "round-trip: recupera os degraus exatos" begin
        ref = Date(2026, 9, 16)
        # Reuniões fictícias; a Selic passa a vigorar no dia seguinte a cada uma
        mdates = [Date(2026, 11, 4), Date(2026, 12, 9), Date(2027, 1, 27),
                  Date(2027, 3, 17), Date(2027, 4, 28)]
        meetings = fake_meetings(mdates)

        # Trajetória verdadeira: 14,00 e cortes de 25 bps, com uma manutenção
        true_path = [14.00, 13.75, 13.50, 13.50, 13.00, 12.75]
        bounds = vcat(ref, [d + Day(1) for d in mdates])

        zr = step_curve(ref, bounds, true_path)
        df = implied_selic_path(zr, ref, meetings; current_rate = true_path[1])

        # O período corrente tem de devolver a Selic vigente
        @test metadata(df, "spot_period") ≈ true_path[1] atol = 1e-8

        # E cada reunião, a taxa do período que ela inaugura
        @test nrow(df) == length(mdates) - 1
        @test df.implied_selic ≈ true_path[2:length(mdates)] atol = 1e-8

        # Movimentos em bps
        @test df.move_bps ≈ [-25.0, -25.0, 0.0, -50.0] atol = 1e-6
        @test df.n_cuts_25 ≈ [-1.0, -1.0, 0.0, -2.0] atol = 1e-6

        # Acumulado contra a Selic corrente
        @test df.cumulative_bps ≈ [-25.0, -50.0, -50.0, -100.0] atol = 1e-6
    end

    @testset "curva plana não precifica movimento" begin
        ref = Date(2026, 9, 16)
        mdates = [Date(2026, 11, 4), Date(2026, 12, 9), Date(2027, 1, 27), Date(2027, 3, 17)]
        meetings = fake_meetings(mdates)

        flat = 13.5
        r_cont = log(1 + flat / 100)
        df = implied_selic_path(t -> r_cont, ref, meetings; current_rate = flat)

        @test all(abs.(df.move_bps) .< 1e-8)
        @test all(isapprox.(df.implied_selic, flat; atol = 1e-8))
        @test metadata(df, "spot_period") ≈ flat atol = 1e-8
    end

    @testset "altas saem com sinal positivo" begin
        ref = Date(2026, 9, 16)
        mdates = [Date(2026, 11, 4), Date(2026, 12, 9), Date(2027, 1, 27)]
        meetings = fake_meetings(mdates)
        true_path = [10.00, 10.50, 11.00, 11.00]
        bounds = vcat(ref, [d + Day(1) for d in mdates])

        df = implied_selic_path(step_curve(ref, bounds, true_path), ref, meetings;
                                current_rate = true_path[1])
        @test df.move_bps ≈ [50.0, 50.0] atol = 1e-6
        @test df.n_cuts_25 ≈ [2.0, 2.0] atol = 1e-6
    end

    @testset "a data de vigência é o dia seguinte à decisão" begin
        m = CopomMeeting(Date(2026, 8, 5), 280, 14.0, false, false)
        @test effective_date(m) == Date(2026, 8, 6)
    end

    @testset "calendário real" begin
        cal = load_copom_calendar()
        @test length(cal) > 280
        @test issorted([m.date for m in cal])

        # Toda reunião ordinária desde 2015 cai numa quarta-feira
        recentes = filter(m -> m.date >= Date(2015, 1, 1) && !m.extraordinary, cal)
        @test all(m -> dayofweek(m.date) == 3, recentes)

        # Selic vigente numa data conhecida: a decidida em 05/08/2026 (14,00%)
        @test current_selic(cal, Date(2026, 9, 1)) ≈ 14.0
        # Na véspera da vigência ainda vale a anterior
        @test current_selic(cal, Date(2026, 8, 5)) ≈ 14.25

        # Próximas reuniões a partir de uma data no meio do calendário
        prox = next_meetings(cal, Date(2026, 9, 1); n = 3)
        @test [m.date for m in prox] == [Date(2026, 9, 16), Date(2026, 11, 4), Date(2026, 12, 9)]
    end

    @testset "validação de entradas" begin
        ref = Date(2026, 9, 16)
        meetings = fake_meetings([Date(2026, 11, 4), Date(2026, 12, 9)])
        flat = t -> log(1.135)

        @test_throws ArgumentError implied_selic_path(flat, ref, CopomMeeting[])

        # Reunião anterior à data de referência não pode entrar
        passadas = fake_meetings([Date(2026, 1, 28), Date(2026, 11, 4)])
        @test_throws ArgumentError implied_selic_path(flat, ref, passadas)

        # Fora de ordem
        fora = fake_meetings([Date(2026, 12, 9), Date(2026, 11, 4)])
        @test_throws ArgumentError implied_selic_path(flat, ref, fora)
    end
end
