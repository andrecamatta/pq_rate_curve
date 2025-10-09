"""
config_service.jl - Centralized configuration management service

This module provides a single source of truth for configuration access,
implementing the Dependency Inversion Principle (DIP) and simplifying testing.
"""

"""
    ConfigService

Encapsulates configuration data loaded from TOML files.
Provides typed accessor methods for different configuration sections.
"""
struct ConfigService
    data::Dict{String,Any}
end

"""
    load_config(path::String="config.toml") -> ConfigService

Load configuration from a TOML file.

# Arguments
- `path`: Path to TOML configuration file (default: "config.toml")

# Returns
- ConfigService instance with loaded configuration data
"""
function load_config(path::String="config.toml")
    ConfigService(TOML.parsefile(path))
end

"""
    get_cv_config(cs::ConfigService) -> Dict{String,Any}

Get cross-validation configuration section.
"""
function get_cv_config(cs::ConfigService)
    get(cs.data, "cross_validation", Dict())
end

"""
    get_pso_config(cs::ConfigService) -> Dict{String,Any}

Get PSO (Particle Swarm Optimization) configuration section.
"""
function get_pso_config(cs::ConfigService)
    get(cs.data, "pso", Dict())
end

"""
    get_hyperparams_config(cs::ConfigService) -> Dict{String,Any}

Get hyperparameter search configuration section.
"""
function get_hyperparams_config(cs::ConfigService)
    get(cs.data, "hyperparameter_search", Dict())
end

"""
    get_raw_config(cs::ConfigService) -> Dict{String,Any}

Get the raw configuration dictionary.
Useful when you need access to the entire config or custom sections.
"""
function get_raw_config(cs::ConfigService)
    cs.data
end

# Global instance for default config (lazy-loaded singleton pattern)
const _DEFAULT_CONFIG = Ref{Union{ConfigService,Nothing}}(nothing)

"""
    default_config() -> ConfigService

Get the default ConfigService instance (loads from "config.toml").
Uses lazy-loading singleton pattern for efficiency.
"""
function default_config()
    if _DEFAULT_CONFIG[] === nothing
        _DEFAULT_CONFIG[] = load_config()
    end
    return _DEFAULT_CONFIG[]
end

"""
    reset_default_config!()

Reset the default config cache. Useful for testing or when config.toml changes.
"""
function reset_default_config!()
    _DEFAULT_CONFIG[] = nothing
end
