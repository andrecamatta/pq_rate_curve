"""
financial_math.jl - Pure financial mathematics functions

This module contains all pure mathematical functions for financial calculations,
including yield curve modeling, bond pricing, and duration calculations.
No side effects or I/O operations.
"""

using Dates, Statistics, BusinessDays

"""
    yearfrac(start_date::Date, end_date::Date) -> Float64

Calculate the year fraction between two dates using ACT/365 convention.
"""
function yearfrac(start_date::Date, end_date::Date; cal=BusinessDays.BRSettlement())
    return BusinessDays.bdayscount(cal, start_date, end_date) / 252.0
end

"""
    nss_rate(t::Float64, params::Vector{Float64}) -> Float64

Calculate Nelson-Siegel-Svensson interest rate for time `t` in years.

Parameters:
- t: Time to maturity in years
- params: Vector [β₀, β₁, β₂, β₃, τ₁, τ₂] containing NSS parameters

Returns:
- Interest rate as a decimal (e.g., 0.10 for 10%)
"""
function nss_rate(t, p)
    # Evita divisão por zero e garante positividade dos taus
    tau1 = max(abs(p[5]), 0.005)
    tau2 = max(abs(p[6]), 0.005)
    
    # Componentes Nelson-Siegel originais
    lambda1 = (1 - exp(-t/tau1)) / (t/tau1)
    lambda2 = lambda1 - exp(-t/tau1)
    
    # Componente Svensson adicional
    lambda3 = (1 - exp(-t/tau2)) / (t/tau2) - exp(-t/tau2)
    
    # Soma ponderada dos componentes
    rate = p[1] + p[2] * lambda1 + p[3] * lambda2 + p[4] * lambda3

    if isnan(rate) || isinf(rate)
        return 1e9 # Retorna um valor alto para penalizar
    end

    return rate
end

"""
    price_bond(cash_flow::Vector, ref_date::Date, params::Vector{Float64}) -> Float64

Price a bond using Nelson-Siegel-Svensson yield curve.

Parameters:
- cash_flow: Vector of (date, amount) tuples representing bond cash flows
- ref_date: Reference date for pricing
- params: NSS parameters vector

Returns:
- Bond price in currency units
"""
function price_bond(cash_flow, ref_date, params)
    price = 0.0
    for (date, amount) in cash_flow
        t = yearfrac(ref_date, date)
        if t > 0  # Apenas fluxos futuros
            rate = nss_rate(t, params)
            price += amount / (1 + rate)^t
        end
    end
    return price
end

"""
    price_bond_precalc(cash_flow_with_times::Vector, params::Vector{Float64}) -> Float64

Price a bond using pre-calculated time fractions for performance optimization.

Parameters:
- cash_flow_with_times: Vector of (time_fraction, amount) tuples
- params: NSS parameters vector

Returns:
- Bond price in currency units
"""
function price_bond_precalc(cash_flow_with_times, params)
    price = 0.0
    for (t, amount) in cash_flow_with_times
        if t > 0
            rate = nss_rate(t, params)
            price += amount / (1 + rate)^t
        end
    end
    return price
end

"""
    calculate_duration(cash_flow::Vector, ref_date::Date, params::Vector{Float64}) -> Float64

Calculate modified duration of a bond using NSS yield curve.

Parameters:
- cash_flow: Vector of (date, amount) tuples representing bond cash flows
- ref_date: Reference date for calculation
- params: NSS parameters vector

Returns:
- Modified duration in years
"""
function calculate_duration(cash_flow, ref_date, params)
    bond_price = price_bond(cash_flow, ref_date, params)
    
    if bond_price <= 0
        return 0.0
    end
    
    duration = 0.0
    for (date, amount) in cash_flow
        t = yearfrac(ref_date, date)
        if t > 0
            rate = nss_rate(t, params)
            pv = amount / (1 + rate)^t
            duration += (t * pv) / bond_price
        end
    end
    
    return duration
end

"""
    calculate_duration_precalc(cash_flow_with_times::Vector, params::Vector{Float64}) -> Float64

Calculate modified duration using pre-calculated time fractions for performance.

Parameters:
- cash_flow_with_times: Vector of (time_fraction, amount) tuples
- params: NSS parameters vector

Returns:
- Modified duration in years
"""
function calculate_duration_precalc(cash_flow_with_times, params)
    bond_price = price_bond_precalc(cash_flow_with_times, params)
    
    if bond_price <= 0
        return 0.0
    end
    
    duration = 0.0
    for (t, amount) in cash_flow_with_times
        if t > 0
            rate = nss_rate(t, params)
            pv = amount / (1 + rate)^t
            duration += (t * pv) / bond_price
        end
    end
    
    return duration
end

"""
    calculate_ytm(market_price::Float64, cash_flow::Vector, ref_date::Date) -> Float64

Calculate yield to maturity (TIR) of a bond given its market price.

Parameters:
- market_price: Market price of the bond
- cash_flow: Vector of (date, amount) tuples representing bond cash flows
- ref_date: Reference date for calculation

Returns:
- Yield to maturity as a decimal
"""
function calculate_ytm(market_price, cash_flow, ref_date)
    # For zero-coupon bonds (LTN), direct calculation
    if length(cash_flow) == 1
        maturity_date, face_value = cash_flow[1]
        t = yearfrac(ref_date, maturity_date)
        return (face_value / market_price)^(1/t) - 1
    end
    
    # For coupon bonds, use iterative method
    function bond_price(ytm)
        price = 0.0
        for (date, amount) in cash_flow
            t = yearfrac(ref_date, date)
            price += amount / (1 + ytm)^t
        end
        return price
    end
    
    # Bisection method to find YTM
    ytm_low = 0.001   # 0.1%
    ytm_high = 0.50   # 50%
    
    for _ in 1:100  # maximum 100 iterations
        ytm_mid = (ytm_low + ytm_high) / 2
        price_mid = bond_price(ytm_mid)
        
        if abs(price_mid - market_price) < 0.001  # tolerance of R$ 0.001
            return ytm_mid
        end
        
        if price_mid > market_price
            ytm_low = ytm_mid
        else
            ytm_high = ytm_mid
        end
    end
    
    return (ytm_low + ytm_high) / 2
end

"""
    calculate_model_ytm(cash_flow::Vector, ref_date::Date, params::Vector{Float64}) -> Float64

Calculate the YTM that the NSS model produces for a bond.

Parameters:
- cash_flow: Vector of (date, amount) tuples representing bond cash flows
- ref_date: Reference date for calculation
- params: NSS parameters vector

Returns:
- Model-implied yield to maturity as a decimal
"""
function calculate_model_ytm(cash_flow, ref_date, params)
    # For simple bonds (LTN), YTM is the spot rate for time to maturity
    if length(cash_flow) == 1
        t = yearfrac(ref_date, cash_flow[1][1])
        return nss_rate(t, params)
    end
    
    # For coupon bonds, calculate YTM iteratively
    model_price = price_bond(cash_flow, ref_date, params)
    return calculate_ytm(model_price, cash_flow, ref_date)
end

function validate_discount_factors(params; max_maturity_years=30.0)
    # Verificar apenas pontos críticos, não 300 pontos
    time_points = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0]

    for t in time_points
        rate = nss_rate(t, params)
        
        # Permitir taxas levemente negativas (até -2%) e limitar superiormente
        if rate < -0.02 || rate > 1.0  # Mais realista que a restrição anterior
            return false
        end
        
        # Verificar apenas se é número válido
        if isnan(rate) || isinf(rate)
            return false
        end
    end

    return true
end