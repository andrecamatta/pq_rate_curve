#!/usr/bin/env julia
using SQLite, Dates, DataFrames

db = SQLite.DB("historical_curves.db")

# Conta registros
result = DBInterface.execute(db, "SELECT COUNT(*) as count FROM nss_curves")
count = first(collect(result)).count
println("Total de registros: $count")

# Mostra primeiros 3
if count > 0
    result = DBInterface.execute(db, "SELECT * FROM nss_curves LIMIT 3")
    df = DataFrame(result)
    println("\nPrimeiros 3 registros:")
    println(df)
end
