#!/usr/bin/env julia --project=..

"""
basic_usage.jl - Exemplo básico de uso do módulo PQRateCurve

Este script demonstra como usar o módulo PQRateCurve para:
1. Importar o módulo
2. Calcular taxas NSS
3. Precificar títulos
4. Calcular duration
"""

using PQRateCurve
using Dates

println("=" ^ 60)
println("Exemplo de Uso do Módulo PQRateCurve")
println("=" ^ 60)

# 1. Parâmetros NSS de exemplo (β0, β1, β2, β3, τ1, τ2)
params = [0.10, -0.02, -0.01, 0.005, 5.0, 15.0]
println("\n📊 Parâmetros NSS:")
println("   β0 = $(params[1])")
println("   β1 = $(params[2])")
println("   β2 = $(params[3])")
println("   β3 = $(params[4])")
println("   τ1 = $(params[5])")
println("   τ2 = $(params[6])")

# 2. Calcular taxas para diferentes prazos
println("\n📈 Taxas NSS para diferentes maturidades:")
maturities = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
for t in maturities
    rate = nss_rate(t, params)
    println("   $t anos: $(round(rate * 100, digits=2))% a.a.")
end

# 3. Exemplo de precificação de título LTN (zero-coupon)
println("\n💰 Exemplo: Precificação de LTN")
ref_date = Date(2024, 1, 15)
maturity_date = Date(2025, 1, 15)
face_value = 1000.0

# Cash flow: apenas o valor de face no vencimento
cash_flow_ltn = [(maturity_date, face_value)]

# Calcular preço teórico
theoretical_price = price_bond(cash_flow_ltn, ref_date, params)
println("   Data de referência: $ref_date")
println("   Vencimento: $maturity_date")
println("   Valor de face: R\$ $(face_value)")
println("   Preço teórico: R\$ $(round(theoretical_price, digits=2))")

# 4. Exemplo de precificação de NTN-F (cupom semestral)
println("\n💰 Exemplo: Precificação de NTN-F")
ntnf_maturity = Date(2027, 1, 1)

# Gerar cash flows semestrais (cupom de 10% a.a. = 5% semestral)
cash_flow_ntnf = Tuple{Date, Float64}[]
let current_date = ref_date + Month(6)
    while current_date < ntnf_maturity
        push!(cash_flow_ntnf, (current_date, 50.0))  # Cupom semestral
        current_date += Month(6)
    end
end
push!(cash_flow_ntnf, (ntnf_maturity, 1050.0))  # Último cupom + principal

theoretical_price_ntnf = price_bond(cash_flow_ntnf, ref_date, params)
println("   Data de referência: $ref_date")
println("   Vencimento: $ntnf_maturity")
println("   Cupons semestrais: R\$ 50.00")
println("   Preço teórico: R\$ $(round(theoretical_price_ntnf, digits=2))")

# 5. Calcular duration
duration_ntnf = calculate_duration(cash_flow_ntnf, ref_date, params)
println("   Duration (Macaulay): $(round(duration_ntnf, digits=2)) anos")

# 6. Calcular YTM
ytm = calculate_ytm(theoretical_price_ntnf, cash_flow_ntnf, ref_date)
println("   YTM: $(round(ytm * 100, digits=2))% a.a.")

println("\n✅ Exemplo concluído com sucesso!")
println("=" ^ 60)
