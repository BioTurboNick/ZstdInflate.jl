# ============================================================
# Section 5: Huffman decode table
#   Reference: RFC 8878 §4.2
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

Base.@propagate_inbounds getindex(ht::HuffmanTable, args...) = getindex(ht.decode_table, args...)

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
    for idx ∈ eachindex(decode_table) .- 1
        entry = decode_table[idx + 1]
        nbits_remaining = L - entry.stream_nbits
        nbits_remaining > 0 || continue
        idx2 = (idx << entry.stream_nbits) & (length(decode_table) - 1)
        nbits_sym2 = Int(nbits_sym1s[idx2 + 1])
        nbits_sym2 ≤ nbits_remaining || continue
        entry2 = decode_table[idx2 + 1]
        decode_table[idx + 1] = HuffmanTableEntry{L}((entry.symbols[1], entry2.symbols[1]), UInt8(entry.stream_nbits + nbits_sym2), 0x02)
    end

    return HuffmanTable{L, typeof(decode_table)}(decode_table)
end

# Decode one Huffman symbol — caller must ensure nbits ≥ max_bits (no refill).
@inline function _huffman_decode_nocheck!(rb::ReverseBitReader, ht::HuffmanTable{L}) where L
    idx = _shr(rb.bits, 64 - L) % Int
    ht_entry = @inbounds ht.decode_table[idx + 1]
    nb1 = Int(ht_entry.stream_nbits)
    rb.bits = _shl(rb.bits, nb1)
    rb.nbits -= nb1
    return Int(ht_entry.symbols[1])
end

# Decode 1 or 2 Huffman symbols into out[p] (and out[p+1] if nsymbols==2 && p<p_end).
# Refills the bit reader as needed. Returns the number of symbols written.
@inline function _huffman_decode12!(rb::ReverseBitReader, ht::HuffmanTable{L},
                                    out::Vector{UInt8}, p::Int, p_end::Int) where L
    rb.nbits < L && refill!(rb)
    idx      = _shr(rb.bits, 64 - L) % Int
    e        = @inbounds ht.decode_table[idx + 1]
    nb_total = Int(e.stream_nbits)
    @inbounds out[p] = e.symbols[1]
    rb.bits   = _shl(rb.bits, nb_total)
    rb.nbits -= nb_total
    if e.nsymbols > 1 && p < p_end
        @inbounds out[p + 1] = e.symbols[2]
        return 2
    end
    return 1
end

# Decode up to 2 Huffman symbols — caller must ensure nbits ≥ max_bits (no refill).
# Writes sym1 and sym2 to out[pos] and out[pos+1] unconditionally (caller must
# ensure out has one byte of slack past regen_size for the last-slot case).
# Returns the number of symbols decoded (1 or 2).
@inline function _huffman_decode2_nocheck!(rb::ReverseBitReader, ht::HuffmanTable{L},
                                           out::Vector{UInt8}, pos::Int) where L
    idx = _shr(rb.bits, 64 - L) % Int
    ht_entry = @inbounds ht.decode_table[idx + 1]
    nb_total = Int(ht_entry.stream_nbits)
    @inbounds out[pos:pos + 1] .= ht_entry.symbols
    rb.bits = _shl(rb.bits, nb_total)
    rb.nbits -= nb_total
    return ht_entry.nsymbols
end


# Decode two symbols from each of 2 streams simultaneously using SIMD.
# Mirrors _huffman_decode4x_2nocheck! but operates on ReverseBitReader2X.
@inline function _huffman_decode2x_2nocheck!(rb::ReverseBitReader2X, ht::HuffmanTable{L},
                                              out::Vector{UInt8},
                                              oi::Vec{2, Int}) where L
    idx = _shr.(rb.bits, Int64(64 - L))

    @inbounds e = (
        ht.decode_table[idx[1] % Int + 1],
        ht.decode_table[idx[2] % Int + 1]
    )

    GC.@preserve out unsafe_store!.(Ptr{NTuple{2, UInt8}}.(pointer.(Ref(out), (oi[1], oi[2]))),
                                    htol.(getfield.(e, :symbols)))

    nb_total = (Int64(e[1].stream_nbits), Int64(e[2].stream_nbits))
    rb.bits  = _shl.(rb.bits, nb_total)
    rb.nbits = rb.nbits .- nb_total

    return Vec{2, UInt32}((UInt32(e[1].nsymbols), UInt32(e[2].nsymbols)))
end

# Decode two symbols from each of 4 streams simultaneously using SIMD.
# bits/nbits from ReverseBitReader4X are loaded into Vec{4,...} for the
# index extraction and bit-consumption steps; table lookups remain scalar.
# Writes symbol pairs to out at o1..o4 (unconditionally — caller ensures slack).
# Returns (adv1, adv2, adv3, adv4): 1 or 2 symbols advanced per stream.
@inline function _huffman_decode4x_2nocheck!(rb::ReverseBitReader4X, ht::HuffmanTable{L},
                                              out::Vector{UInt8},
                                              oi::Vec{4, Int}) where L
    idx = _shr.(rb.bits, Int64(64 - L))

    @inbounds e = (
        ht.decode_table[idx[1] % Int + 1],
        ht.decode_table[idx[2] % Int + 1],
        ht.decode_table[idx[3] % Int + 1],
        ht.decode_table[idx[4] % Int + 1]
    )

    # Write symbol pairs as a 16-bit store each: sym1 at out[oi], sym2 at out[oi+1].
    # In the single-symbol case sym2 lands one slot past the actual end — safe because
    # the caller guarantees at least one byte of slack past each segment boundary.
    GC.@preserve out unsafe_store!.(Ptr{NTuple{2, UInt8}}.(pointer.(Ref(out), Tuple(oi))),
                                    htol.(getfield.(e, :symbols)))

    nb_total = (Int64(e[1].stream_nbits), Int64(e[2].stream_nbits),
                Int64(e[3].stream_nbits), Int64(e[4].stream_nbits))
    rb.bits  = _shl.(rb.bits, nb_total)
    rb.nbits = rb.nbits .- nb_total

    return Vec{4, UInt32}((UInt32(e[1].nsymbols), UInt32(e[2].nsymbols),
                           UInt32(e[3].nsymbols), UInt32(e[4].nsymbols)))
end


# ============================================================
# Section 6: Huffman tree description
#   Reference: RFC 8878 §4.2.1
# ============================================================

# Infer the weight of the last symbol given the weight array so far.
# Returns the (last_sym, last_weight) pair, where last_sym is the index
# of the symbol whose weight we must fill in.
function _infer_last_weight(weights::Vector{UInt8})
    # RFC 8878 §4.2.1: sum of 2^(w-1) for all present symbols must be a
    # power of two (call it P). The last (highest) symbol gets w = log2(P - sum) + 1.
    # Returns (last_sym_index, last_weight, table_log).
    total = UInt64(0)
    for w in weights
        w == 0 && continue
        total += UInt64(1) << (Int(w) - 1)
    end
    total == 0 && return (length(weights), 1, 1)  # single symbol edge case
    # tableLog = floor(log2(total)) + 1  (reference: BIT_highbit32(weightTotal) + 1)
    table_log = _flog2(Int(total)) + 1
    p = UInt64(1) << table_log
    p > total || (table_log += 1; p <<= 1)   # ensure p > total
    rest = Int(p - total)
    rest > 0 && (rest & (rest - 1)) == 0 || throw(ArgumentError("zstd: invalid Huffman weight sum"))
    last_w = _flog2(rest) + 1
    return (length(weights) + 1, last_w, table_log)
end

# Decode FSE-compressed Huffman weights.
# Uses two interleaved FSE states that are updated alternately.
# RFC 8878 §4.2.1.1
function _decode_fse_weights(br::ForwardBitReader, byte_limit::Int)
    # Read the FSE table for weights
    al, dist  = read_fse_dist!(br, HUFTABLE_LOG_MAX)
    t         = build_fse_table(dist, al)

    # The remainder of the weight description is a reverse bitstream
    pos_after = byte_pos(br)
    n_remain  = byte_limit - pos_after + 1
    n_remain > 0 || throw(ArgumentError("zstd: no data for Huffman weight FSE stream"))

    rb = ReverseBitReader(@view br.data[pos_after:pos_after+n_remain-1])

    # Init two interleaved states
    state1 = fse_init!(rb, t)
    state2 = fse_init!(rb, t)

    weights = UInt8[]
    sizehint!(weights, 256)
    # Interleaved FSE decode matching the reference zstd tail loop.
    # Each iteration: decode symbol (peek + update), emit it, then
    # check if the stream overflowed.  After overflow, emit the OTHER
    # state's pending symbol and break.
    while true
        sym1 = fse_peek(t, state1)
        state1 = _fse_update_unchecked(rb, t, state1)
        push!(weights, UInt8(sym1))
        if _rbr_overflowed(rb) || length(weights) ≥ 255
            push!(weights, UInt8(fse_peek(t, state2)))
            break
        end

        sym2 = fse_peek(t, state2)
        state2 = _fse_update_unchecked(rb, t, state2)
        push!(weights, UInt8(sym2))
        if _rbr_overflowed(rb) || length(weights) ≥ 255
            push!(weights, UInt8(fse_peek(t, state1)))
            break
        end
    end

    # Advance the forward reader past the weight data
    br.pos   = byte_limit + 1
    br.nbits = 0
    br.bits  = UInt64(0)

    return weights
end

# Read the Huffman tree description and return a HuffmanTable.
# data[pos] is the first byte of the description.
# Returns (HuffmanTable, bytes_consumed).
function read_huffman_description(data::Vector{UInt8}, pos::Int)
    header = Int(data[pos])
    if header < 128
        # FSE-compressed weights: header = compressed_size
        compressed_size = header
        # Forward bit reader over the compressed bytes
        br = ForwardBitReader(@view data[pos+1:pos+compressed_size])
        weights = _decode_fse_weights(br, compressed_size)
        last_sym, last_w, table_log = _infer_last_weight(weights)
        push!(weights, UInt8(last_w))
        ht = build_huffman_table!(weights, table_log)
        return ht, compressed_size + 1
    else
        # Direct representation: (header - 127) weight nibbles follow
        nsyms = header - 127
        # Each byte holds two nibbles; ceil(nsyms/2) bytes
        nbytes = (nsyms + 1) >> 1
        weights = Vector{UInt8}(undef, nsyms)
        for i in 1:nbytes
            b = data[pos + i]
            lo = b & 0x0f
            hi = (b >> 4) & 0x0f
            idx = (i-1)*2
            if idx + 1 ≤ nsyms
                weights[idx + 1] = hi    # high nibble = first weight
            end
            if idx + 2 ≤ nsyms
                weights[idx + 2] = lo    # low nibble = second weight
            end
        end
        last_sym, last_w, table_log = _infer_last_weight(weights)
        if last_sym ≤ nsyms
            weights[last_sym] = UInt8(last_w)
        else
            push!(weights, UInt8(last_w))
        end
        ht = build_huffman_table!(weights, table_log)
        return ht, nbytes + 1
    end
end
