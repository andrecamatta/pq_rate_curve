"""
formatting.jl - Formatting utilities for consistent output across the codebase

This module centralizes all formatting logic for numbers, providing a single source
of truth for precision and display formats. This ensures consistency and makes it
easy to adjust precision globally.
"""

"""
    format_percentage(x::Real) -> String

Format a decimal value as a percentage with 2 decimal places.

# Example
```julia
format_percentage(0.1234) # "12.34%"
```
"""
format_percentage(x::Real) = string(round(x * 100, digits=2), "%")

"""
    format_score(x::Real) -> Float64

Format a score value with 6 decimal places.
Used for cost functions, hybrid scores, etc.
"""
format_score(x::Real) = round(x, digits=6)

"""
    format_cost(x::Real) -> Float64

Format a cost value with 6 decimal places.
Alias for format_score for semantic clarity.
"""
format_cost(x::Real) = round(x, digits=6)

"""
    format_basis_points(x::Real) -> Float64

Format a value in basis points with 1 decimal place.
Used for vertex stability, rate variations, etc.
"""
format_basis_points(x::Real) = round(x, digits=1)

"""
    format_nss_params(params::Vector{Float64}) -> Vector{Float64}

Format NSS parameters with appropriate precision:
- β₀, β₁, β₂, β₃: 4 decimal places
- τ₁, τ₂: 2 decimal places

# Example
```julia
format_nss_params([0.123456, -0.054321, 0.012345, -0.006789, 5.123456, 12.654321])
# [0.1235, -0.0543, 0.0123, -0.0068, 5.12, 12.65]
```
"""
function format_nss_params(params::Vector{Float64})
    if length(params) != 6
        error("NSS parameters must have exactly 6 elements [β₀, β₁, β₂, β₃, τ₁, τ₂]")
    end

    return [
        round(params[1], digits=4),  # β₀
        round(params[2], digits=4),  # β₁
        round(params[3], digits=4),  # β₂
        round(params[4], digits=4),  # β₃
        round(params[5], digits=2),  # τ₁
        round(params[6], digits=2)   # τ₂
    ]
end

"""
    format_pso_coefficient(x::Real) -> Float64

Format PSO coefficients (C1, C2, ω) with 2 decimal places.
"""
format_pso_coefficient(x::Real) = round(x, digits=2)

"""
    format_temporal_penalty(x::Real) -> Float64

Format temporal penalty weight with 4 decimal places.
"""
format_temporal_penalty(x::Real) = round(x, digits=4)

"""
    format_overfitting_ratio(x::Real) -> Float64

Format overfitting ratio with 3 decimal places.
"""
format_overfitting_ratio(x::Real) = round(x, digits=3)

"""
    format_time_minutes(seconds::Real) -> Float64

Convert seconds to minutes and format with 1 decimal place.
"""
format_time_minutes(seconds::Real) = round(seconds / 60, digits=1)

"""
    format_time_seconds(seconds::Real) -> Float64

Format time in seconds with 1 decimal place.
"""
format_time_seconds(seconds::Real) = round(seconds, digits=1)

"""
    format_currency(x::Real) -> Float64

Format currency values (bond prices, errors in R\$) with 2 decimal places.
"""
format_currency(x::Real) = round(x, digits=2)

"""
    format_quantity(x::Real) -> Float64

Format trading quantities with 0 decimal places (whole numbers).
"""
format_quantity(x::Real) = round(x, digits=0)

"""
    format_error_threshold(x::Real) -> Float64

Format error thresholds with 2 decimal places.
"""
format_error_threshold(x::Real) = round(x, digits=2)

"""
    format_liquidity_factor(x::Real) -> Float64

Format liquidity factors with 4 decimal places.
"""
format_liquidity_factor(x::Real) = round(x, digits=4)
