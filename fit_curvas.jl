#!/usr/bin/env julia --project=.

# Script unificado para fit de curvas NSS
# Uso: julia --project=. fit_curvas.jl [--start YYYY-MM-DD] [--end YYYY-MM-DD] [opções]
# Ou: ./fit_curvas.jl [--start YYYY-MM-DD] [--end YYYY-MM-DD] [opções]

using PQRateCurve
using Dates, ArgParse

# Parse command line arguments
function parse_commandline()
    s = ArgParseSettings(
        description = "Fit Nelson-Siegel-Svensson yield curves for a date range",
        epilog = "Example: julia --project=. fit_curvas.jl --start 2024-01-01 --end 2024-12-31"
    )

    @add_arg_table! s begin
        "--start"
            help = "Start date (YYYY-MM-DD)"
            default = "2024-01-01"
        "--end"
            help = "End date (YYYY-MM-DD)"
            default = "2024-12-31"
        "--output"
            help = "Output CSV base filename"
            default = "curvas_nss"
        "--continuity"
            help = "Enable temporal continuity (search for previous parameters)"
            action = :store_true
            default = true
        "--no-continuity"
            help = "Disable temporal continuity"
            action = :store_false
            dest_name = "continuity"
        "--verbose"
            help = "Print detailed progress"
            action = :store_true
            default = true
        "--quiet"
            help = "Suppress progress output"
            action = :store_false
            dest_name = "verbose"
        "--config"
            help = "Path to configuration file"
            default = "config.toml"
    end

    return parse_args(s)
end

function main()
    args = parse_commandline()

    # Parse dates
    try
        start_date = Date(args["start"])
        end_date = Date(args["end"])

        if start_date > end_date
            println("❌ Erro: Data inicial não pode ser posterior à data final!")
            exit(1)
        end

        if end_date > today()
            println("⚠️ Data final é no futuro, usando hoje como limite")
            end_date = today()
        end

        # Call high-level API function
        results, config = fit_curves_for_period(
            start_date, end_date;
            config_file=args["config"],
            output_csv=args["output"],
            find_continuity=args["continuity"],
            verbose=args["verbose"]
        )

        # Summary already printed by fit_curves_for_period
        if !isempty(results)
            successful = sum(r.success for r in results)
            if args["verbose"]
                println("\\n🎉 Fit concluído! Sucesso: $successful/$(length(results))")
            end
        end

    catch e
        if isa(e, ArgumentError) && contains(string(e), "invalid Date")
            println("❌ Formato de data inválido. Use YYYY-MM-DD")
            exit(1)
        else
            println("❌ Erro: $e")
            rethrow()
        end
    end
end

# Execute if called directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
