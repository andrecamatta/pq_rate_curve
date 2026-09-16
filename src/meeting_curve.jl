"""
meeting_curve.jl - Curva com degraus nas datas de reunião do Copom

O oposto de interpolar. Em vez de ajustar uma função suave à curva inteira e
derivá-la para obter forwards — o que amplifica erro de ajuste e, pior, apaga
por construção os degraus que queremos medir — aqui a estrutura de degrau é
**imposta**: a taxa forward é constante por pedaço e só muda nas datas de
vigência do Copom. Os níveis dos degraus são resolvidos para reprecificar os
instrumentos observados.

Com isso os degraus *são* a resposta: cada nível é a Selic que o mercado
precifica para o período entre duas reuniões.

Duas propriedades importantes:

1. **A ponta curta não é extrapolada.** O primeiro degrau, do dia de referência
   até a primeira vigência, é a Selic corrente — que é conhecida. Ele entra
   como restrição, não como incógnita, o que elimina a região abaixo do
   título mais curto negociado.

2. **Movimento não identificado não é inventado.** Quando há menos instrumentos
   que reuniões o sistema é subdeterminado; entre todas as soluções que
   reprecificam os títulos igualmente bem, escolhe-se a de menor movimento
   total entre degraus consecutivos. O modelo não atribui corte onde o dado
   não sustenta — ele espalha.

Convenção brasileira em tudo: taxa efetiva anual sobre 252 dias úteis.
"""

# ============================================================================
# Instrumentos
# ============================================================================

"""
    ZeroObservation

Uma taxa zero observada: o prazo de vencimento e a taxa efetiva anual (% a.a.,
base 252). Uma LTN gera uma diretamente do preço, por ser zero-cupom:

    i = (1000 / PU)^(252/du) - 1
"""
struct ZeroObservation
    maturity::Date
    rate::Float64        # % a.a. efetiva, base 252
    quantity::Float64    # volume negociado, usado como peso
end

"""
    ltn_zero_rates(df, ref_date) -> Vector{ZeroObservation}

Extrai taxas zero das LTNs negociadas na data. Não há ajuste de curva no
caminho: a LTN é zero-cupom, então o preço médio observado dá a taxa do
vencimento diretamente.
"""
function ltn_zero_rates(df::DataFrame, ref_date::Date)
    obs = ZeroObservation[]
    isempty(df) && return obs

    for row in eachrow(df)
        row.bond_code == "LTN" || continue
        du = BusinessDays.bdayscount(BusinessDays.BRSettlement(), ref_date, row.maturity_date)
        du > 0 || continue
        price = row.avg_price
        (price > 0 && price < LTN_FACE_VALUE * 2) || continue

        rate = (LTN_FACE_VALUE / price)^(252 / du) - 1
        qty = hasproperty(df, :quantity_traded) && !ismissing(row.quantity_traded) ?
              Float64(row.quantity_traded) : 1.0
        push!(obs, ZeroObservation(row.maturity_date, rate * 100, qty))
    end

    sort!(obs, by = o -> o.maturity)
    return obs
end

# ============================================================================
# Curva de degraus
# ============================================================================

"""
    MeetingCurve

Curva com forward constante entre vigências do Copom.

- `segment_start[k]`: início do k-ésimo período (o 1º é a data de referência,
  os demais são datas de vigência)
- `forward[k]`: taxa efetiva anual (% a.a.) que vigora nesse período — é a
  Selic precificada
- `fit_error_bps[j]`: erro de reprecificação do instrumento `j`, em pontos-base
  de taxa
- `identified[k]`: se há instrumento vencendo depois do início do período `k`
  que o discipline individualmente
"""
struct MeetingCurve
    ref_date::Date
    meetings::Vector{CopomMeeting}
    segment_start::Vector{Date}
    forward::Vector{Float64}
    observations::Vector{ZeroObservation}
    fit_error_bps::Vector{Float64}
    identified::Vector{Bool}
    current_rate::Float64
end

function Base.show(io::IO, mc::MeetingCurve)
    n_id = count(mc.identified)
    print(io, "MeetingCurve(", mc.ref_date, ", ", length(mc.forward), " degraus, ",
          n_id, " identificados, ", length(mc.observations), " instrumentos, ",
          "erro máx ", round(maximum(abs, mc.fit_error_bps; init = 0.0), digits = 1), " bps)")
end

"""
    _segment_daycounts(ref_date, bounds, maturity) -> Vector{Float64}

Anos (base 252) que cada segmento contribui para o desconto até `maturity`.
`bounds` são os inícios dos segmentos; o último se estende indefinidamente.
"""
function _segment_daycounts(bounds::Vector{Date}, maturity::Date)
    cal = BusinessDays.BRSettlement()
    K = length(bounds)
    out = zeros(K)
    for k in 1:K
        seg_start = bounds[k]
        seg_end = k < K ? bounds[k+1] : maturity
        s = min(seg_start, maturity)
        e = min(seg_end, maturity)
        e > s && (out[k] = BusinessDays.bdayscount(cal, s, e) / 252)
    end
    return out
end

"""
    bootstrap_meeting_curve(ref_date, observations, meetings; current_rate,
                            smoothness=1e-6) -> MeetingCurve

Resolve os degraus da curva a partir das taxas zero observadas.

Cada instrumento impõe

    Σₖ fₖ · duₖ/252 = log(1 + iⱼ) · duⱼ/252

com `fₖ = log(1 + degrauₖ)` a forward contínua do segmento `k`. O primeiro
degrau é fixado na Selic corrente e sai do sistema.

`smoothness` pondera a penalidade sobre diferenças entre degraus consecutivos.
Deve ficar pequeno: seu papel não é suavizar o resultado, e sim escolher, entre
as soluções que reprecificam os títulos igualmente bem, aquela que não inventa
movimento onde o dado não identifica. Aumentar demais achata degraus reais.
"""
function bootstrap_meeting_curve(ref_date::Date,
                                 observations::Vector{ZeroObservation},
                                 meetings::Vector{CopomMeeting};
                                 current_rate::Float64,
                                 smoothness::Float64 = 1e-6)

    isempty(observations) && throw(ArgumentError("Nenhum instrumento observado em $ref_date"))
    isempty(meetings) && throw(ArgumentError("Nenhuma reunião fornecida"))
    all(m -> effective_date(m) > ref_date, meetings) ||
        throw(ArgumentError("Há reuniões cuja vigência não é posterior a $ref_date"))

    bounds = vcat(ref_date, [effective_date(m) for m in meetings])
    K = length(bounds)                       # número de degraus

    # Só instrumentos que vencem depois da data de referência
    obs = filter(o -> o.maturity > ref_date, observations)
    isempty(obs) && throw(ArgumentError("Nenhum instrumento vencendo após $ref_date"))
    N = length(obs)

    # Sistema A·f = y, com f em forward contínua
    A = Matrix{Float64}(undef, N, K)
    y = Vector{Float64}(undef, N)
    cal = BusinessDays.BRSettlement()
    for (j, o) in enumerate(obs)
        A[j, :] = _segment_daycounts(bounds, o.maturity)
        du = BusinessDays.bdayscount(cal, ref_date, o.maturity)
        y[j] = log(1 + o.rate / 100) * du / 252
    end

    # O primeiro degrau é a Selic corrente: conhecido, não estimado
    f0 = log(1 + current_rate / 100)
    y_adj = y .- A[:, 1] .* f0
    A_free = A[:, 2:end]
    Kf = K - 1

    # Peso por liquidez, normalizado
    w = [sqrt(max(o.quantity, 1.0)) for o in obs]
    w ./= maximum(w)
    Aw = A_free .* w
    yw = y_adj .* w

    # Penalidade sobre diferenças entre degraus consecutivos (inclui o salto do
    # degrau conhecido para o primeiro estimado)
    D = zeros(Kf, Kf)
    for k in 1:Kf
        D[k, k] = 1.0
        k > 1 && (D[k, k-1] = -1.0)
    end
    d0 = zeros(Kf); d0[1] = f0          # 1ª diferença é contra o degrau fixado

    scale = maximum(abs, Aw) + eps()
    λ = smoothness * scale

    # Mínimos quadrados aumentado: [Aw; λD] f = [yw; λd0]
    M = vcat(Aw, λ .* D)
    b = vcat(yw, λ .* d0)
    f_free = M \ b

    f = vcat(f0, f_free)
    forward = (exp.(f) .- 1) .* 100

    # Erro de reprecificação, convertido para taxa do instrumento
    resid = A * f .- y
    fit_error_bps = Vector{Float64}(undef, N)
    for (j, o) in enumerate(obs)
        du = BusinessDays.bdayscount(cal, ref_date, o.maturity)
        implied = exp((y[j] + resid[j]) * 252 / du) - 1
        fit_error_bps[j] = (implied - o.rate / 100) * 10_000
    end

    # Um degrau só é identificado se algum instrumento vence dentro ou depois
    # dele — caso contrário seu nível vem inteiramente da penalidade
    identified = [any(o -> o.maturity > bounds[k], obs) for k in 1:K]

    return MeetingCurve(ref_date, meetings, bounds, forward, obs,
                        fit_error_bps, identified, current_rate)
end

# ============================================================================
# Saídas
# ============================================================================

"""
    implied_path(mc::MeetingCurve) -> DataFrame

Trajetória de Selic precificada, reunião a reunião.

Colunas: `meeting_date`, `meeting_number`, `implied_selic`, `move_bps`,
`cumulative_bps`, `n_cuts_25`, `identified`.

`identified = false` marca reunião cujo nível não é sustentado por nenhum
instrumento — o valor vem da regularização e não deve ser lido como preço de
mercado.
"""
function implied_path(mc::MeetingCurve)
    n = length(mc.meetings)
    df = DataFrame(meeting_date = Date[], meeting_number = Int[],
                   implied_selic = Float64[], move_bps = Float64[],
                   cumulative_bps = Float64[], n_cuts_25 = Float64[],
                   identified = Bool[])
    for i in 1:n
        rate = mc.forward[i+1]
        prev = mc.forward[i]
        push!(df, (mc.meetings[i].date, mc.meetings[i].number, rate,
                   (rate - prev) * 100, (rate - mc.current_rate) * 100,
                   (rate - prev) * 100 / 25, mc.identified[i+1]))
    end
    return df
end

"""
    zero_rate_curve(mc::MeetingCurve) -> Function

Devolve `t -> taxa zero contínua anualizada`, para uso onde se espera uma curva
(por exemplo comparação com o NSS). Reconstrói o desconto acumulando os degraus.
"""
function zero_rate_curve(mc::MeetingCurve)
    cal = BusinessDays.BRSettlement()
    f = log.(1 .+ mc.forward ./ 100)
    K = length(f)
    # dias úteis acumulados até o início de cada segmento
    du_bounds = [BusinessDays.bdayscount(cal, mc.ref_date, d) for d in mc.segment_start]

    return function (t)
        t <= 0 && return f[1]
        target = t * 252
        acc, done = 0.0, 0.0
        for k in 1:K
            seg = k < K ? (du_bounds[k+1] - du_bounds[k]) : Inf
            take = min(Float64(seg), target - done)
            take <= 0 && break
            acc += take / 252 * f[k]
            done += take
            done >= target - 1e-9 && break
        end
        return acc / t
    end
end
