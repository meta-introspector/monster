#!/usr/bin/env julia
"""Apply harmonic analysis to register values"""

using FFTW
using LinearAlgebra
using JSON

function analyze_registers(input_file, output_file)
    println("📊 Loading registers from: $input_file")
    
    # Load register data
    data = JSON.parsefile(input_file)
    
    results = []
    
    # Extract register sequences
    reg_names = ["AX", "BX", "CX", "DX", "SI", "DI", "R8", "R9", "R10", "R11", "R12", "R13", "R14", "R15"]
    
    for reg in reg_names
        # Extract values for this register
        values = Float64[]
        for entry in data
            if haskey(entry["regs"], reg)
                push!(values, Float64(entry["regs"][reg]))
            end
        end
        
        if length(values) < 2
            continue
        end
        
        # Apply FFT
        fft_result = fft(values)
        
        # Compute power spectrum
        power = abs2.(fft_result)
        
        # Find dominant frequencies
        sorted_indices = sortperm(power, rev=true)
        top_freqs = sorted_indices[1:min(10, length(sorted_indices))]
        
        # Store results
        push!(results, Dict(
            "register" => reg,
            "count" => length(values),
            "mean" => mean(values),
            "std" => std(values),
            "fft_size" => length(fft_result),
            "top_frequencies" => top_freqs,
            "top_powers" => power[top_freqs],
            "total_power" => sum(power)
        ))
        
        println("  ✓ $reg: $(length(values)) samples, $(length(top_freqs)) dominant frequencies")
    end
    
    # Save to JSON (will be converted to parquet by Python)
    open(output_file * ".json", "w") do f
        JSON.print(f, results, 2)
    end
    
    println("✅ Harmonic analysis complete: $(length(results)) registers analyzed")
end

if length(ARGS) < 2
    println("Usage: harmonic_analysis.jl <input.json> <output.parquet>")
    exit(1)
end

analyze_registers(ARGS[1], ARGS[2])
