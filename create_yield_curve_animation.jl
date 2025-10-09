#!/usr/bin/env julia --project=.
# create_yield_curve_animation.jl
# Creates an animated video of yield curves from the CSV output with NSS parameters
# Usage: julia --project=. create_yield_curve_animation.jl [input_csv_file] [output_video.mp4]

using PQRateCurve
using Dates, TOML

# Parse command line arguments
function parse_args()
    if length(ARGS) < 1
        println("❌ Error: Please provide the input CSV file as the first argument")
        println("Usage: julia --project=. create_yield_curve_animation.jl input_csv_file [output_video.mp4]")
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

# Main execution
input_file, output_video = parse_args()

# Load config if available (for animation parameters)
config_file = isfile("config.toml") ? "config.toml" : nothing

# Call the high-level API function from the module
try
    result_path = create_yield_curve_animation(
        input_file,
        output_video;
        config_file=config_file
    )

    println("[ Info: Saved animation to $result_path")
catch e
    println("❌ Error creating animation: $e")
    exit(1)
end
