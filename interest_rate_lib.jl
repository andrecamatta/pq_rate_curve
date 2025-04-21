using Dates
using Printf
using InterestRates
using BusinessDays 
using Logging 
using Optim  # Adicionado
using DataFrames # Adicionado

const DAYS_IN_YEAR = 252.0  # Dias úteis no Brasil (convenção ANBIMA)
const NOMINAL_ANNUAL_COUPON_RATE = 0.10 # Taxa nominal anual da NTN-F
const FACE_VALUE = 1000.0
const BR_CAL = BusinessDays.BRSettlement() # Adicionado aqui

# Fontes de dados para os preços dos títulos
@enum DataSource tesouro_direto_manha=1 bacen_avg_price=2

# Representa um título brasileiro prefixado (LTN ou NTN-F)
struct BRLBond
    name::String        # Nome/Tipo do título
    maturity::Date      # Data de vencimento
    price::Float64      # Preço unitário (PU)
    has_coupon::Bool    # true para NTN-F, false para LTN
    
    # Construtor interno para converter string de data DD/MM/YYYY
    function BRLBond(name::String, maturity_str::AbstractString, price::Float64, has_coupon::Bool)
        parts = split(maturity_str, "/")
        day_str, month_str, year_str = parts
        maturity = Date(parse(Int, year_str), parse(Int, month_str), parse(Int, day_str))
        new(name, maturity, price, has_coupon)
    end
end

# Formata uma data para o formato DD/MM/YYYY
format_date(date::Date) = Dates.format(date, "dd/mm/yyyy")

# Verifica se um título tem cupom baseado no nome
has_coupon(bond_name::String) = occursin("Juros Semestrais", bond_name) # Adicionado

# Converte string DD/MM/YYYY para Date
to_date(date_str::AbstractString, format::String="d/m/Y") = Date(date_str, dateformat"d/m/Y") # Adicionado

yearfrac(start_date::String, end_date::String) = 
    BusinessDays.bdayscount(BR_CAL, to_date(start_date), to_date(end_date)) / DAYS_IN_YEAR

function generate_coupon_dates(start_date::String, maturity_date::String)
    maturity = to_date(maturity_date)
    start = to_date(start_date)
    
    @assert Dates.day(maturity) ∈ [1, 15] "Data de vencimento deve ser dia 1 ou 15"
    
    coupon_dates = maturity:-Month(6):start |> collect |> reverse
    format_date.(coupon_dates)
end

# Calcula fluxo de caixa
function calculate_cash_flow(face_value::Float64, annual_rate::Float64, 
    start_date::String, maturity_date::String, is_bullet::Bool=false)
is_bullet && return [(maturity_date, face_value)]

semiannual_rate = (1 + annual_rate)^(1/2) - 1
coupon_dates = generate_coupon_dates(start_date, maturity_date)
coupon_value = face_value * semiannual_rate

[(date, i == length(coupon_dates) ? coupon_value + face_value : coupon_value) 
for (i, date) in enumerate(coupon_dates)]
end

# Calcula taxa implícita
function calculate_implied_rate(pu::Float64, pu_date::String, cash_flow::Vector)
    pu_dt = to_date(pu_date)
    
    objective(rate) = begin
        daily_rate = (1 + rate)^(1/DAYS_IN_YEAR) - 1
        sum(cash_flow) do (date, value)
            days = BusinessDays.bdayscount(BR_CAL, pu_dt, to_date(date))
            value / (1 + daily_rate)^days
        end |> x -> abs(x - pu)
    end
    
    result = optimize(objective, 0.0, 1.0, Brent()) # Usa optimize de Optim
    !Optim.converged(result) && error("Não foi possível convergir para taxa implícita") # Usa Optim.converged
    
    Optim.minimizer(result) # Usa Optim.minimizer
end

# Constrói curva zero-coupon
function build_zero_coupon_curve(zero_coupon_bonds::Vector{BRLBond}, ref_date::String)
    zero_coupon_curve = Dict{Date, Float64}()
    for bond in zero_coupon_bonds
        # Corrigido: Acessa o campo correto 'price' em vez de 'pu'
        yield = calculate_implied_rate(bond.price, ref_date, [(format_date(bond.maturity), FACE_VALUE)]) 
        zero_coupon_curve[bond.maturity] = yield # Corrigido: Usa Date como chave
    end
    return zero_coupon_curve
end

# Ajusta curva iterativamente usando títulos com cupom
function iterative_curve_adjustment(
    zero_coupon_curve::Dict{Date, Float64},
    coupon_bonds::Vector{BRLBond},
    ref_date::String
)
    adjusted_curve = deepcopy(zero_coupon_curve)
    
    for bond in coupon_bonds
        cash_flow = calculate_cash_flow(FACE_VALUE, NOMINAL_ANNUAL_COUPON_RATE, ref_date, format_date(bond.maturity), false) 
        observed_price = bond.price # Corrigido: Acessa o campo correto
        
        function objective(rate)
            adjusted_curve[bond.maturity] = rate # Corrigido: Usa Date como chave
            price = price_with_coupon(cash_flow, adjusted_curve, ref_date)
            return abs(price - observed_price)
        end
        
        result = optimize(objective, 0.0, 1.0, Brent()) # Usa optimize de Optim
        new_rate = Optim.minimizer(result) # Usa Optim.minimizer
        adjusted_curve[bond.maturity] = new_rate # Corrigido: Usa Date como chave
    end
    
    return adjusted_curve
end

# Constrói curva completa usando todos os títulos
function build_complete_yield_curve(bonds::Vector{BRLBond}, ref_date::String)
    zero_coupon_bonds = filter(b -> !b.has_coupon, bonds)
    coupon_bonds = filter(b -> b.has_coupon, bonds)
    
    zero_coupon_curve = build_zero_coupon_curve(zero_coupon_bonds, ref_date)
    complete_curve = iterative_curve_adjustment(zero_coupon_curve, coupon_bonds, ref_date)
    
    ref_dt = to_date(ref_date)
    dates = collect(keys(complete_curve)) |> sort
    days = [BusinessDays.bdayscount(BR_CAL, ref_dt, d) for d in dates]
    rates = [complete_curve[d] for d in dates]
    
    curve = InterestRates.IRCurve("curva-completa",
        InterestRates.BDays252(:Brazil),
        InterestRates.ExponentialCompounding(),
        InterestRates.FlatForward(),
        ref_dt,
        days,
        rates)
    
    daily_curve = Dict{Date, Float64}()
    current_date = first(dates)
    last_date = last(dates)
    
    while current_date <= last_date
        if BusinessDays.isbday(BR_CAL, current_date)
            rate = InterestRates.zero_rate(curve, current_date)
            daily_curve[current_date] = rate
        end
        current_date = current_date + Day(1)
    end
    
    return daily_curve
end

# Cria curva interpolada
function create_curve(zero_coupon_curve::Dict{Date, Float64}, ref_date::Date)
    dates = collect(keys(zero_coupon_curve)) |> sort
    rates = [zero_coupon_curve[d] for d in dates]
    days = [BusinessDays.bdayscount(BR_CAL, ref_date, d) for d in dates]
    
    InterestRates.IRCurve("curva-zero",
        InterestRates.BDays252(:Brazil),
        InterestRates.ExponentialCompounding(),
        InterestRates.FlatForward(),
        ref_date,
        days,
        rates)
end

# Precifica título com cupom usando curva interpolada
function price_with_coupon(cash_flow::Vector{Tuple{String, Float64}}, zero_coupon_curve::Dict{Date, Float64}, ref_date::String)
    ref_dt = to_date(ref_date)
    curve = create_curve(zero_coupon_curve, ref_dt)
    
    price = 0.0
    for (date_str, value) in cash_flow
        date = to_date(date_str)
        df = InterestRates.discountfactor(curve, date)
        price += value * df
    end
    return price
end

# Compara duas curvas (Dict{Date, Float64}) para termos específicos (em anos).
# Retorna um DataFrame com as taxas (em %) para cada termo e cada curva.
function compare_curves(
    curve1_dict::Dict{Date, Float64}, 
    curve2_dict::Dict{Date, Float64}, 
    ref_date::String, 
    terms::Vector{Float64}=collect(0.5:0.5:10.0); # Termos padrão: 0.5 a 10 anos
    curve1_name::String="Curva 1",
    curve2_name::String="Curva 2"
)
    ref_dt = to_date(ref_date)
    
    # Cria objetos IRCurve
    ir_curve1 = create_curve(curve1_dict, ref_dt)
    ir_curve2 = create_curve(curve2_dict, ref_dt)
    
    rates1_vec = Union{Missing, Float64}[]
    rates2_vec = Union{Missing, Float64}[]
    
    # Calcula taxa para cada termo usando InterestRates.zero_rate
    for term in terms
        # Calcula a data futura correspondente ao termo em anos, usando dias úteis
        days_to_advance = round(Int, term * DAYS_IN_YEAR) # Converte anos para dias úteis
        target_date = BusinessDays.advancebdays(BR_CAL, ref_dt, days_to_advance)
        
        rate1 = try
            # Obtém a taxa zero (spot) para a data alvo
            InterestRates.zero_rate(ir_curve1, target_date) * 100  # Converte para %
        catch e 
            @warn "Erro ao calcular taxa para '$curve1_name' no termo $term (data $target_date): $e" 
            missing 
        end
        push!(rates1_vec, rate1)
        
        rate2 = try
            # Obtém a taxa zero (spot) para a data alvo
            InterestRates.zero_rate(ir_curve2, target_date) * 100  # Converte para %
        catch e 
            @warn "Erro ao calcular taxa para '$curve2_name' no termo $term (data $target_date): $e" 
            missing
        end
        push!(rates2_vec, rate2)
    end

    # Monta o DataFrame de resultados
    result = DataFrame( # Usa DataFrame de DataFrames
        "Termo (anos)" => terms,
        curve1_name => rates1_vec,
        curve2_name => rates2_vec
    )
    
    return result
end