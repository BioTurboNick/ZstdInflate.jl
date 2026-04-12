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
        new{L}(symbols, steam_nbits, nsymbols)
    end
    function HuffmanTableEntry{L}() where L
        new{L}((0x00, 0x00), 0x00, 0x00)
    end
end

struct HuffmanTable{L}
    decode_table::Vector{HuffmanTableEntry{L}}
end

@propagate_inbounds getindex(ht::HuffmanTable, args...) = getindex(ht.decode_table, args)

# Build a dual-symbol Huffman decode table from a weight array.
# weights[i+1] = weight for symbol i (0 = absent; weight w ≥ 1 means code length
# max_bits - w + 1, probability 2^(w-1)).
# Each entry is a UInt32: (nb_total<<24)|(nb1<<16)|(sym2<<8)|sym1.
function build_huffman_table(weights::Vector{UInt8}, max_bits::Int)
    nsyms = length(weights)
    max_bits > 0 ||
        throw(ArgumentError("zstd: all-zero Huffman weights"))
    max_bits ≤ HUFTABLE_LOG_MAX ||
        throw(ArgumentError("zstd: Huffman table log $max_bits exceeds maximum ($HUFTABLE_LOG_MAX)"))

    table_size = 1 << max_bits

    decode_table = HuffmanTable{max_bits}(fill(HuffmanTableEntry{max_bits}(), table_size))

    # Pass 1: Populate single-symbol entries for all symbols
    rank_count = zeros(Int, max_bits + 1)
    next_rank_start = zeros(Int, max_bits + 1)
    for sym in 0:nsyms - 1
        w = Int(weights[sym + 1])
        w > 0 && (rank_count[w] += 1)
    end
    pos = 0
    for w in 1:max_bits
        next_rank_start[w] = pos
        pos += rank_count[w] * (1 << (w - 1))
    end
    for sym in UInt8.(0:nsyms - 1)
        w = Int(weights[sym + 1])
        w == 0 && continue
        nb1 = UInt8(max_bits - w + 1)
        num_entries = 1 << (w - 1)
        entry = HuffmanTableEntry{max_bits}((sym, 0x00), nb1, 0x01)
        start = next_rank_start[w]
        for j in 0:num_entries - 1
            decode_table[start + j + 1] = entry
        end
        next_rank_start[w] += num_entries
    end

    # Pass 2: Add second symbols where there is room in the entry (i.e., nbits1 + nbits2 ≤ max_bits)
    for idx in 0:table_size - 1
        entry = decode_table[idx + 1]
        nbits_remaining = max_bits - entry.stream_nbits
        nbits_remaining > 0 || continue
        idx2 = (idx << entry.stream_nbits) & (table_size - 1)
        (sym2, nbits_sym2) = decode_table[idx2 + 1]
        nbits_sym2 ≤ nbits_remaining || continue
        decode_table[idx + 1] = HuffmanTableEntry{max_bits}((entry.symbols[1], sym2), entry.stream_nbits + nbits_sym2, 0x02)
    end

    return HuffmanTable{max_bits}(decode_table)
end

# Decode one Huffman symbol — caller must ensure nbits ≥ max_bits (no refill).
@inline function _huffman_decode_nocheck!(rb::ReverseBitReader, ht::HuffmanTable{L}) where L
    idx = _shr(rb.bits, 64 - L) % Int
    ht_entry = @inbounds ht[idx + 1]
    nb1 = Int(ht_entry.stream_nbits)
    rb.bits = _shl(rb.bits, nb1)
    rb.nbits -= nb1
    return Int(ht_entry.symbols[1])
end

# Decode one Huffman symbol from the reverse bit reader.
@inline function huffman_decode!(rb::ReverseBitReader, ht::HuffmanTable{L}) where L
    rb.nbits < L && refill!(rb)
    _huffman_decode_nocheck!(rb, ht)
end

# Decode up to 2 Huffman symbols — caller must ensure nbits ≥ max_bits (no refill).
# Writes sym1 and sym2 to out[pos] and out[pos+1] unconditionally (caller must
# ensure out has one byte of slack past regen_size for the last-slot case).
# Returns the number of symbols decoded (1 or 2).
@inline function _huffman_decode2_nocheck!(rb::ReverseBitReader, ht::HuffmanTable{L},
                                           out::Vector{UInt8}, pos::Int) where L
    idx = _shr(rb.bits, 64 - L) % Int
    ht_entry = @inbounds ht[idx + 1]
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

    GC.@preserve out unsafe_store!.(Ptr{UInt16}.(pointer.(Ref(out), (oi[1], oi[2]))),
                                    htol.(e .% UInt16))

    nb_total = Int64.(e .>> 24)
    rb.bits  = _shl.(rb.bits, nb_total)
    rb.nbits = rb.nbits .- nb_total

    e_vec = Vec{2, UInt32}(e)
    nb1   = (e_vec >> UInt32(16)) & Vec{2, UInt32}(UInt32(0xFF))
    return min((e_vec >> UInt32(24)) - nb1, Vec{2, UInt32}(UInt32(1))) + Vec{2, UInt32}(UInt32(1))
end

# Decode two symbols from each of 4 streams simultaneously using SIMD.
# bits/nbits from ReverseBitReader4X are loaded into Vec{4,...} for the
# index extraction and bit-consumption steps; table lookups remain scalar.
# Writes symbol pairs to out at o1..o4 (unconditionally — caller ensures slack).
# Returns (adv1, adv2, adv3, adv4): 1 or 2 symbols advanced per stream.
@inline function _huffman_decode4x_2nocheck!(rb::ReverseBitReader4X, ht::HuffmanTable{L},
                                              out::Vector{UInt8},
                                              oi::Vec{4, Int}) where L
    # Extract all 4 table indices in parallel: shift MSBs down to top L bits.
    idx = _shr.(rb.bits, Int64(64 - L))

    # Scalar table lookups (gather not supported in SIMD.jl).
    # @inbounds: idx[i] is in [0, 2^L) by construction (top L bits of a UInt64),
    # so idx[i] % Int + 1 is in [1, 2^L] = valid range for decode_table.
    @inbounds e = (
        ht.decode_table[idx[1] % Int + 1],
        ht.decode_table[idx[2] % Int + 1],
        ht.decode_table[idx[3] % Int + 1],
        ht.decode_table[idx[4] % Int + 1]
    )

    # Write symbol pairs as a single 16-bit store each: e[i] & 0xFFFF = (sym2<<8)|sym1,
    # which on a little-endian machine writes sym1 at out[oi] and sym2 at out[oi+1].
    # In the single-symbol case sym2 lands one slot past the actual end — safe because
    # the caller guarantees at least one byte of slack past each segment boundary.
    GC.@preserve out unsafe_store!.(Ptr{UInt16}.(pointer.(Ref(out), Tuple(oi))), htol.(e .% UInt16))

    # Consume bits and update counts.
    # _shl (masks shift count with 63) avoids the dead safety check Julia's `<<` emits for
    # shifts ≥ 64: nb_total ≤ L ≤ 11, so the check is unreachable but LLVM can't prove it.
    nb_total = Int64.(e .>> 24)
    rb.bits  = _shl.(rb.bits, nb_total)
    rb.nbits = rb.nbits .- nb_total

    # Advance: 2 if double-symbol entry (nb_total > nb1), else 1.
    # nb_total - nb1 is 0 for single, ≥1 for double; min(..., 1) + 1 gives 1 or 2.
    # Compiles to vpminud + vpaddd — 2 insns replacing 31 scalar ones.
    e_vec = Vec{4, UInt32}(e)
    nb1   = (e_vec >> UInt32(16)) & Vec{4, UInt32}(UInt32(0xFF))
    return min((e_vec >> UInt32(24)) - nb1, Vec{4, UInt32}(UInt32(1))) + Vec{4, UInt32}(UInt32(1))
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
        ht = build_huffman_table(weights, table_log)
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
        ht = build_huffman_table(weights, table_log)
        return ht, nbytes + 1
    end
end
