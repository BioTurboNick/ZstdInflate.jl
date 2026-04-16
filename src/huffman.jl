# ============================================================
# Huffman decode table
#   Reference: RFC 8878 §4.2.1.3
# ============================================================

#=
The Huffman decode table is a flat array of 2^L entries, where `L` is the maximum code length of any symbol.
The index into the table is the next `L` read from the bitstream. Each entry contains either one or two symbols,
the number of bits of the stream they take up, and the number of symbols present. If nsymbols == 1, the second
symbol entry is invalid. The design allows us to opportunistically read a second symbol in the same read.
=#

struct HuffmanTableEntry{L} # TODO: The type param may not be needed, test it
    symbols::NTuple{2, UInt8} # Second symbol is valid only if nbits_total > nbits_sym1
    stream_nbits::UInt8       # [0:L]
    nsymbols::UInt8           # 0, 1 or 2 (0 should only be present during table construction; 0 means invalid entry) TODO: Check if this is true

    function HuffmanTableEntry{L}(symbols::NTuple{2, UInt8}, stream_nbits::UInt8, nsymbols::UInt8) where L
        stream_nbits ≤ L ||
            throw(ArgumentError("zstd: Huffman table entry stream_nbits $stream_nbits exceeds max_bits ($L)"))
        0 ≤ nsymbols ≤ 2 ||
            throw(ArgumentError("zstd: Huffman table entry nsymbols must be 0, 1 or 2"))
        new{L}(symbols, stream_nbits, nsymbols)
    end
    function HuffmanTableEntry{L}() where L
        new{L}((0x00, 0x00), 0x00, 0x00)
    end
end

struct HuffmanTable{L, T <: AbstractVector{HuffmanTableEntry{L}}}
    decode_table::T
end

Base.@propagate_inbounds Base.getindex(ht::HuffmanTable, args...) = getindex(ht.decode_table, args...)

# Build a dual-symbol Huffman decode table from a weight array.
function build_huffman_table!(weights::Vector{UInt8}, max_bits::Int; kwargs...)
    max_bits > 0 ||
        throw(ArgumentError("zstd: all-zero Huffman weights"))
    max_bits ≤ HUFTABLE_LOG_MAX ||
        throw(ArgumentError("zstd: Huffman table log $max_bits exceeds maximum ($HUFTABLE_LOG_MAX)"))

    table_size = 1 << max_bits
    v = fill(HuffmanTableEntry{max_bits}(), table_size)
    return build_huffman_table!(v, weights; kwargs...)
end

function build_huffman_table!(decode_table::AbstractVector{HuffmanTableEntry{L}}, weights::Vector{UInt8}; scratch_buffers::Union{Nothing, NTuple{2, AbstractVector{Int}}} = nothing) where L
    fill!(decode_table, HuffmanTableEntry{L}())

    # Pass 1: Populate single-symbol entries for all symbols
    if isa(scratch_buffers, NTuple{2, AbstractVector{Int}})
        rank_count = resize!(fill!(scratch_buffers[1], 0x00), L)
        next_rank_start = resize!(fill!(scratch_buffers[2], 0x00), L)
    else
        rank_count = zeros(Int, L)
        next_rank_start = zeros(Int, L)
    end
    for w ∈ weights
        w > 0 || continue
        rank_count[w] += 1
    end
    next_rank_start[2:end] .= cumsum(ntuple(w -> rank_count[w] * (1 << (w - 1)), Val(L - 1)))
    for (i, w) ∈ enumerate(weights)
        w > 0 || continue
        sym = UInt8(i - 1)
        nbits_sym1 = UInt8(L - w + 1)
        entry = HuffmanTableEntry{L}((sym, 0x00), nbits_sym1, 0x01)
        start = next_rank_start[w]
        num_entries = 1 << (w - 1)
        decode_table[start .+ (1:num_entries)] .= Ref(entry)
        next_rank_start[w] += num_entries
    end

    # Pass 2: Add second symbols where there is room in the entry (i.e., nbits1 + nbits2 ≤ max_bits)
    nbits_sym1s = [decode_table[i].stream_nbits for i ∈ eachindex(decode_table)]
    for (i, entry) ∈ enumerate(decode_table)
        code = i - 1
        nbits_remaining = L - entry.stream_nbits
        nbits_remaining > 0 || continue
        code2 = (code << entry.stream_nbits) & (length(decode_table) - 1)
        j = code2 + 1
        nbits_sym2 = Int(nbits_sym1s[j])
        nbits_sym2 ≤ nbits_remaining || continue
        entry2 = decode_table[j]
        stream_nbits = UInt8(entry.stream_nbits + nbits_sym2)
        decode_table[i] = HuffmanTableEntry{L}((entry.symbols[1], entry2.symbols[1]), stream_nbits, 0x02)
    end

    return HuffmanTable{L, typeof(decode_table)}(decode_table)
end
