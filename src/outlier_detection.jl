"""
outlier_detection.jl - Outlier detection functionality

This module contains functions for detecting outliers in bond pricing using
Median Absolute Deviation (MAD) and liquidity criteria.
"""

using Statistics

"""
    calculate_mad(values::Vector{Float64}) -> Float64

Calculate the Median Absolute Deviation (MAD) of a vector of values.
MAD = median(|xi - median(x)|)

Parameters:
- values: Vector of numeric values

Returns:
- MAD value
"""
function calculate_mad(values::Vector{Float64})
    if isempty(values) || length(values) < 2
        return 0.0
    end
    
    median_val = median(values)
    deviations = abs.(values .- median_val)
    return median(deviations)
end

"""
    detect_outliers_mad(cash_flows, ref_date, params; fator_erro=3.0) -> (Vector{Int}, Vector{Float64}, Float64)

Detect outliers based on pricing error using the Median Absolute Deviation (MAD) method.

Parameters:
- cash_flows: List of (market_price, cash_flow) tuples
- ref_date: Reference date
- params: Current NSS parameters
- fator_erro: MAD multiplier for outlier classification (default: 3.0)

Returns:
- outlier_indices: Indices of bonds classified as outliers
- errors: Vector with pricing errors for all bonds
- mad_value: Calculated MAD value
"""
function detect_outliers_mad(cash_flows, ref_date, params; fator_erro=3.0)
    if isempty(cash_flows)
        return Int[], Float64[], 0.0
    end
    
    # Calculate pricing errors for all bonds
    errors = Float64[]
    for (market_price, cash_flow) in cash_flows
        theoretical_price = price_bond(cash_flow, ref_date, params)
        error_abs = abs(theoretical_price - market_price)
        push!(errors, error_abs)
    end
    
    mad_value = calculate_mad(errors)
    # If MAD is zero (e.g., all errors are the same), avoid division by zero or removing everything
    if mad_value < 1e-6
        return Int[], errors, mad_value
    end

    error_threshold = fator_erro * mad_value
    
    # Identify outliers based on error ONLY
    outlier_indices = Int[]
    for (i, error) in enumerate(errors)
        if error > error_threshold
            push!(outlier_indices, i)
        end
    end
    
    return outlier_indices, errors, mad_value
end

"""
    detect_outliers_mad_and_liquidity(cash_flows, bond_quantities, ref_date, params; 
                                     fator_erro=3.0, fator_liq=0.1) -> (Vector{Int}, Vector{Float64}, Float64)

Detect outliers based on TWO simultaneous conditions:
1. Error > fator_erro × MAD  
2. Trading quantity < fator_liq % of total

Parameters:
- cash_flows: List of (market_price, cash_flow) tuples
- bond_quantities: Vector with corresponding trading quantities
- ref_date: Reference date
- params: Current NSS parameters
- fator_erro: MAD multiplier for outlier classification (default: 3.0)
- fator_liq: Percentage of total quantity (default: 0.1 = 10%)

Returns:
- outlier_indices: Indices of bonds classified as outliers
- errors: Vector with pricing errors for all bonds
- mad_value: Calculated MAD value
"""
function detect_outliers_mad_and_liquidity(cash_flows, bond_quantities, ref_date, params; 
                                          fator_erro=3.0, fator_liq=0.1)
    if isempty(cash_flows)
        return Int[], Float64[], 0.0
    end
    
    # Calculate pricing errors for all bonds
    errors = Float64[]
    for (market_price, cash_flow) in cash_flows
        theoretical_price = price_bond(cash_flow, ref_date, params)
        error_abs = abs(theoretical_price - market_price)
        push!(errors, error_abs)
    end
    
    # Calculate MAD of errors
    mad_value = calculate_mad(errors)
    error_threshold = fator_erro * mad_value
    
    # Calculate liquidity threshold
    total_quantity = sum(bond_quantities)
    liquidity_threshold = fator_liq * total_quantity
    
    # Identify outliers that satisfy BOTH conditions
    outlier_indices = Int[]
    for (i, error) in enumerate(errors)
        quantity = bond_quantities[i]
        
        # Condition 1: High error
        high_error = error > error_threshold
        
        # Condition 2: Low liquidity
        low_liquidity = quantity < liquidity_threshold
        
        # Outlier only if BOTH conditions are true
        if high_error && low_liquidity
            push!(outlier_indices, i)
        end
    end
    
    return outlier_indices, errors, mad_value
end

"""
    remove_outliers(cash_flows, bond_quantities, outlier_indices) -> (Vector, Vector)

Remove outliers from cash flows and bond quantities arrays.

Parameters:
- cash_flows: Original cash flows vector
- bond_quantities: Original bond quantities vector
- outlier_indices: Indices of outliers to remove

Returns:
- filtered_cash_flows: Cash flows with outliers removed
- filtered_quantities: Bond quantities with outliers removed
"""
function remove_outliers(cash_flows, bond_quantities, outlier_indices)
    if isempty(outlier_indices)
        return cash_flows, bond_quantities
    end
    
    # Create mask for non-outliers
    n = length(cash_flows)
    keep_mask = trues(n)
    keep_mask[outlier_indices] .= false
    
    # Filter arrays
    filtered_cash_flows = cash_flows[keep_mask]
    filtered_quantities = bond_quantities[keep_mask]
    
    return filtered_cash_flows, filtered_quantities
end

"""
    print_outlier_summary(outlier_indices, errors, mad_value, error_threshold, liquidity_threshold; 
                         cash_flows=nothing, bond_quantities=nothing)

Print a summary of detected outliers for debugging and monitoring.

Parameters:
- outlier_indices: Indices of detected outliers
- errors: All pricing errors
- mad_value: Calculated MAD value
- error_threshold: Error threshold used
- liquidity_threshold: Liquidity threshold used
- cash_flows: Optional cash flows for detailed reporting
- bond_quantities: Optional bond quantities for detailed reporting
"""
function print_outlier_summary(outlier_indices, errors, mad_value, error_threshold, liquidity_threshold; 
                               cash_flows=nothing, bond_quantities=nothing)
    if isempty(outlier_indices)
        println("✅ Nenhum outlier detectado. Processo concluído.")
        return
    end
    
    println("⚠️  Outliers detectados: $(length(outlier_indices)) títulos")
    println("   MAD = $(round(mad_value, digits=3)), Erro threshold = $(round(error_threshold, digits=2)), Liquidez threshold = $(round(liquidity_threshold, digits=1))")
    
    if cash_flows !== nothing && bond_quantities !== nothing
        for i in outlier_indices
            if i <= length(cash_flows) && i <= length(bond_quantities)
                market_price = cash_flows[i][1]
                quantity = bond_quantities[i]
                error = errors[i]
                println("    💥 Erro=R\$$(round(error, digits=2)) | Qtde=$(round(quantity, digits=0)) | Preço=R\$$(round(market_price, digits=2))")
            end
        end
    end
    
    println("🗑️  Removidos $(length(outlier_indices)) outliers. Títulos restantes: $(length(cash_flows) - length(outlier_indices))")
end

"""
    detect_ultra_low_liquidity(bond_quantities; fator_liq=0.015, ultra_low_factor=3.0) -> Vector{Int}

Detect bonds with ultra-low liquidity (independent of pricing error).
This filter removes bonds that are traded in extremely low volumes, which are often:
- Data errors (e.g., volume=1)
- Non-market transactions (accounting adjustments)
- Bonds with unreliable pricing

Parameters:
- bond_quantities: Vector with trading quantities
- fator_liq: Base liquidity percentage threshold (default: 0.015 = 1.5%)
- ultra_low_factor: Divisor for ultra-low threshold (default: 3.0)

Returns:
- outlier_indices: Indices of bonds with ultra-low liquidity

Example:
If fator_liq=0.015 and ultra_low_factor=3.0:
- Normal threshold: 1.5% of total volume
- Ultra-low threshold: 0.5% of total volume (1.5% / 3)
- Bonds with qty < 0.5% of total are removed regardless of error
"""
function detect_ultra_low_liquidity(bond_quantities; fator_liq=0.015, ultra_low_factor=3.0)
    if isempty(bond_quantities)
        return Int[]
    end

    total_quantity = sum(bond_quantities)
    ultra_low_threshold = (fator_liq / ultra_low_factor) * total_quantity

    outlier_indices = Int[]
    for (i, quantity) in enumerate(bond_quantities)
        if quantity < ultra_low_threshold
            push!(outlier_indices, i)
        end
    end

    return outlier_indices
end

"""
    detect_outliers_fixed_threshold(cash_flows, bond_quantities, ref_date, params;
                                    error_threshold_global=20.0, fator_liq=0.015) -> (Vector{Int}, Vector{Float64}, Float64)

Detect outliers using FIXED GLOBAL threshold (instead of MAD-based relative threshold).
This approach is more predictable and robust to extreme outliers.

A bond is classified as outlier if it satisfies BOTH conditions:
1. Absolute error > error_threshold_global (fixed value, e.g., 20.0)
2. Trading quantity < fator_liq % of total volume

Parameters:
- cash_flows: List of (market_price, cash_flow) tuples
- bond_quantities: Vector with corresponding trading quantities
- ref_date: Reference date
- params: Current NSS parameters
- error_threshold_global: Fixed absolute error threshold (default: 20.0)
- fator_liq: Percentage of total quantity for low liquidity (default: 0.015 = 1.5%)

Returns:
- outlier_indices: Indices of bonds classified as outliers
- errors: Vector with pricing errors for all bonds
- mean_error: Mean absolute error (for reporting)

Key differences from MAD-based approach:
- Uses FIXED threshold instead of MAD × factor
- More predictable behavior across different days
- Less sensitive to extreme outliers distorting MAD calculation
"""
function detect_outliers_fixed_threshold(cash_flows, bond_quantities, ref_date, params;
                                        error_threshold_global=20.0, fator_liq=0.015)
    if isempty(cash_flows)
        return Int[], Float64[], 0.0
    end

    # Calculate pricing errors for all bonds
    errors = Float64[]
    for (market_price, cash_flow) in cash_flows
        theoretical_price = price_bond(cash_flow, ref_date, params)
        error_abs = abs(theoretical_price - market_price)
        push!(errors, error_abs)
    end

    # Calculate mean error for reporting
    mean_error = mean(errors)

    # Calculate liquidity threshold
    total_quantity = sum(bond_quantities)
    liquidity_threshold = fator_liq * total_quantity

    # Identify outliers that satisfy BOTH conditions
    outlier_indices = Int[]
    for (i, error) in enumerate(errors)
        quantity = bond_quantities[i]

        # Condition 1: High error (fixed threshold)
        high_error = error > error_threshold_global

        # Condition 2: Low liquidity
        low_liquidity = quantity < liquidity_threshold

        # Outlier only if BOTH conditions are true
        if high_error && low_liquidity
            push!(outlier_indices, i)
        end
    end

    return outlier_indices, errors, mean_error
end