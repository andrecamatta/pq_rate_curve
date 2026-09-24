"""
test_data_refresh.jl - Atualização incremental dos dados e do banco

Cobre três pontos que faziam dias recentes saírem errados ou nunca serem
ajustados:

  - STRIPS de NTN-F (cupom e principal separados) chegam no arquivo do BACEN
    com a mesma SIGLA do título inteiro e têm de ser descartados
  - o arquivo do mês corrente em cache tem de ser rebaixado quando não cobre a
    data pedida
  - datas gravadas como falha voltam como pendentes
"""

using PQRateCurve
using PQRateCurve: clean_bacen_data, cache_is_stale, CacheMetadata, NTNF_STRIP_CODES
using Dates, DataFrames
using Test

@testset "Atualização incremental" begin

    @testset "STRIPS de NTN-F são descartados" begin
        d = Date(2026, 9, 16)
        raw = DataFrame(
            "DATA MOV"        => fill(d, 4),
            "SIGLA"           => ["LTN", "NTN-F", "NTN-F", "NTN-F"],
            "CODIGO"          => [100000, 950199, 950197, 950198],
            "CODIGO ISIN"     => ["BRSTNCLTN", "BRSTNCNTF1P8", "BRSTNCNTF1U8", "BRSTNCNTF2A8"],
            "VENCIMENTO"      => [Date(2027, 1, 1), Date(2027, 1, 1), Date(2027, 1, 1), Date(2033, 1, 1)],
            "PU MED"          => [930.0, 1011.37, 47.08, 430.61],
            "QUANT NEGOCIADA" => [100, 8449, 2761398, 47499],
        )
        df = clean_bacen_data(raw, d, d)

        @test nrow(df) == 2
        @test sort(df.codigo) == [100000, 950199]
        @test all(c -> c ∉ NTNF_STRIP_CODES, df.codigo)
    end

    @testset "cache do mês corrente" begin
        agora = DateTime(2026, 9, 24, 12)
        ym = "202609"
        antigo = CacheMetadata(DateTime(2026, 9, 16, 16), Date(2026, 9, 15), 1, "x")
        recente = CacheMetadata(DateTime(2026, 9, 24, 11), Date(2026, 9, 23), 1, "x")

        # não cobre a data e foi baixado há mais de um dia -> rebaixa
        @test cache_is_stale(ym, antigo, Date(2026, 9, 23); now = agora)
        # cobre a data -> usa o cache
        @test !cache_is_stale(ym, antigo, Date(2026, 9, 15); now = agora)
        # não cobre, mas acabou de ser baixado (dia ainda não publicado) -> usa o cache
        @test !cache_is_stale(ym, recente, Date(2026, 9, 24); now = agora)
        # sem metadado -> rebaixa
        @test cache_is_stale(ym, nothing, Date(2026, 9, 23); now = agora)
        # mês fechado nunca fica velho
        @test !cache_is_stale("202608", antigo, Date(2026, 9, 23); now = agora)
    end

    @testset "falhas voltam como pendentes" begin
        path = tempname() * ".db"
        try
            db = init_database(path)
            d1, d2 = Date(2026, 9, 21), Date(2026, 9, 22)
            save_curve(db, d1, [0.13, 0.0, -0.01, 0.02, 0.5, 5.0], 0.5, 16, 2)
            save_curve_failure(db, d2, "Dados insuficientes (<6 títulos)")

            @test get_missing_dates(db, d1, d2) == [d2]
            @test isempty(get_missing_dates(db, d1, d2; retry_failed = false))
            close(db)
        finally
            for f in (path, path * "-wal", path * "-shm")
                isfile(f) && rm(f; force = true)
            end
        end
    end
end
