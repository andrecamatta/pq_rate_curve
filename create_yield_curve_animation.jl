#!/usr/bin/env julia
# create_yield_curve_animation.jl
# Creates an animated video of yield curves from the CSV output with NSS parameters
# Usage: julia create_yield_curve_animation.jl [input_csv_file] [output_video.mp4]

using CSV, DataFrames, Plots, Dates, Statistics, TOML
using Plots.Measures

# Set plotting backend explicitly
gr()  # Use GR backend for better axis rendering

# Parse command line arguments
function parse_args()
    if length(ARGS) < 1
        println("❌ Error: Please provide the input CSV file as the first argument")
        println("Usage: julia create_yield_curve_animation.jl input_csv_file [output_video.mp4]")
        exit(1)
    end
    
    input_file = ARGS[1]
    
    if length(ARGS) >= 2
        output_video = ARGS[2]
    else
        # Generate output filename with timestamp
        timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
        output_video = "yield_curves_animation_$(timestamp).mp4"
    end
    
    return input_file, output_video
end

# Configuration
input_file, output_video = parse_args()

# Load configuration from config.toml
config = isfile("config.toml") ? TOML.parsefile("config.toml") : Dict()
anim_config = get(config, "animation", Dict())

const FPS = get(anim_config, "fps", 10)
const DURATION = get(anim_config, "duration", 30)
const PLOT_DPI = get(anim_config, "plot_dpi", 200)
const PLOT_SIZE = tuple(get(anim_config, "plot_size", [1200, 800])...)

# NSS yield curve function
function nss_yield(beta0, beta1, beta2, beta3, tau1, tau2, maturity)
    # Ensure we handle both scalar and vector inputs
    if tau1 <= 0 || tau2 <= 0
        return fill(beta0, length(maturity))
    end
    
    rates = similar(maturity)
    for i in eachindex(maturity)
        m = maturity[i]
        if m <= 0
            rates[i] = beta0 + beta1 + beta2 + beta3
        else
            term1 = (1.0 - exp(-m / tau1)) / (m / tau1)
            term2 = (1.0 - exp(-m / tau1)) / (m / tau1) - exp(-m / tau1)
            term3 = (1.0 - exp(-m / tau2)) / (m / tau2) - exp(-m / tau2)
            rates[i] = beta0 + beta1 * term1 + beta2 * term2 + beta3 * term3
        end
    end
    return rates
end

# Load the data
println("📊 Loading yield curve data from $input_file...")
df = CSV.read(input_file, DataFrame)

# Filter successful fits
successful_fits = df[df.Sucesso .== true, :]
dates = successful_fits.Data
sort!(dates)

# Create maturity points
default_maturities = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0]
maturities = get(anim_config, "maturities", default_maturities)

# Create time points for animation
time_points = range(1, length(dates), length=Int(round(FPS * DURATION)))

# Find global min and max rates for consistent y-axis
all_rates = Float64[]
for row in eachrow(successful_fits)
    rates = nss_yield(row.Beta0, row.Beta1, row.Beta2, row.Beta3, row.Tau1, row.Tau2, maturities) .* 100
    append!(all_rates, rates)
end
min_rate = minimum(all_rates)
max_rate = maximum(all_rates)
rate_range = max_rate - min_rate
y_min = max(0, min_rate - 0.1 * rate_range)
y_max = max_rate + 0.1 * rate_range

# Prepare animation
println("🎬 Creating animation...")
anim = @animate for i in time_points
    # Get the closest date index
    idx = min(round(Int, i), length(dates))
    date = dates[idx]
    
    # Get data for this date
    date_data = successful_fits[successful_fits.Data .== date, :][1, :]
    
    # Calculate yield curve
    rates = nss_yield(date_data.Beta0, date_data.Beta1, date_data.Beta2, date_data.Beta3, 
                     date_data.Tau1, date_data.Tau2, maturities) .* 100
    
    # Create the plot with improved x-axis configuration
    p = plot(
        maturities, rates,
        xlabel="Prazo (anos)",
        ylabel="Taxa (%)",
        title="Curvas de Juros - Títulos Públicos Brasileiros",
        legend=false,
        linewidth=3,
        color=:blue,
        marker=:circle,
        markersize=4,
        xlim=(0, 10.2),  # Consistent with xticks range
        ylim=(y_min, y_max),
        dpi=PLOT_DPI,
        size=PLOT_SIZE,
        xticks=(0:1:10, string.(0:1:10)),  # Explicit tick labels
        xtickfontsize=14,
        ytickfontsize=14,
        titlefontsize=18,
        guidefontsize=16,
        bottom_margin=12mm,  # More space for x-axis labels
        left_margin=10mm,    # More space for y-axis labels
        right_margin=8mm,    # More space for annotations
        top_margin=8mm       # Space for title
    )
    
    # Add grid and styling
    plot!(p, xgrid=true, ygrid=true, gridalpha=0.3)
    
    # Add date annotation
    annotate!(p, 8.5, y_max - 0.05 * (y_max - y_min),
              text("$(Dates.format(date, "dd/mm/yyyy"))", 12, :left, color=:black))
    
    # Add month progression indicator (removed English month name)
end

# Save the animation
println("🎥 Saving animation to $output_video...")
mp4(anim, output_video, fps=FPS)

println("✅ Animation complete! Video saved as $output_video")
println("📊 Summary:")
println("   - Input file: $input_file")
println("   - Output video: $output_video")
println("   - Duration: $DURATION seconds")
println("   - FPS: $FPS")
println("   - Total successful curves: $(length(dates))")
println("   - Date range: $(minimum(dates)) to $(maximum(dates))")
println("   - First successful date: $(dates[1])")
println("   - Last successful date: $(dates[end])")