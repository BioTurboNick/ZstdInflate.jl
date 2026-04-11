# ============================================================
# Section 5: Huffman decode table
#   Reference: RFC 8878 §4.2
# ============================================================

struct HuffmanTable{L}
    # Packed dual-symbol decode table indexed by the top L bits of the peeked code word.
    # Each UInt32 entry encodes up to two symbols:
    #   bits [31:24]  nb_total — total bits consumed (nb1 + nb2; ≤ L)
    #   bits [23:16]  nb1      — bits consumed by sym1 alone (needed for single-symbol path)
    #   bits [15: 8]  sym2     — second symbol (valid only when nb_total > nb1)
    #   bits [ 7: 0]  sym1     — first symbol (always valid)
    # nb_total == nb1 indicates a single-symbol entry; nb_total > nb1 is a two-symbol entry.
    decode_table::Vector{UInt32}
end

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
    # Pass 1: build a temporary single-symbol table (sym, nb1) to use as input
    # for the dual-symbol pass below.
    tmp = fill(UInt16(max_bits), table_size)  # (sym<<8)|nb1

    rank_count      = zeros(Int, max_bits + 1)
    next_rank_start = zeros(Int, max_bits + 1)
    for sym in 0:nsyms-1
        w = Int(weights[sym+1])
        w > 0 && (rank_count[w] += 1)
    end
    pos = 0
    for w in 1:max_bits
        next_rank_start[w] = pos
        pos += rank_count[w] * (1 << (w - 1))
    end
    for sym in 0:nsyms-1
        w = Int(weights[sym+1])
        w == 0 && continue
        nb1         = max_bits - w + 1
        num_entries = 1 << (w - 1)
        entry       = UInt16((sym << 8) | nb1)
        start       = next_rank_start[w]
        for j in 0:num_entries-1
            tmp[start + j + 1] = entry
        end
        next_rank_start[w] += num_entries
    end

    # Pass 2: build the full max_bits-wide dual-symbol decode table.
    dtable = Vector{UInt32}(undef, table_size)
    for idx in 0:table_size-1
        e1   = Int(tmp[idx + 1])
        sym1 = (e1 >> 8) & 0xFF
        nb1  = e1 & 0xFF
        rem  = max_bits - nb1   # bits remaining after sym1
        if rem > 0
            idx2 = (idx << nb1) & (table_size - 1)
            e2   = Int(tmp[idx2 + 1])
            sym2 = (e2 >> 8) & 0xFF
            nb2  = e2 & 0xFF
            if nb2 ≤ rem
                dtable[idx + 1] = UInt32((nb1 + nb2) << 24) | UInt32(nb1 << 16) |
                                  UInt32(sym2 << 8) | UInt32(sym1)
                continue
            end
        end
        dtable[idx + 1] = UInt32(nb1 << 24) | UInt32(nb1 << 16) | UInt32(sym1)
    end

    return HuffmanTable{max_bits}(dtable)
end

# Decode one Huffman symbol — caller must ensure nbits ≥ max_bits (no refill).
@inline function _huffman_decode_nocheck!(rb::ReverseBitReader, ht::HuffmanTable{L}) where L
    idx      = _shr(rb.bits, 64 - L) % Int
    e        = @inbounds ht.decode_table[idx + 1]
    nb1      = Int((e >> 16) & 0xFF)
    rb.bits  = _shl(rb.bits, nb1)
    rb.nbits -= nb1
    return Int(e & 0xFF)
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
    idx      = _shr(rb.bits, 64 - L) % Int
    e        = @inbounds ht.decode_table[idx + 1]
    nb_total = Int(e >> 24)
    nb1      = Int((e >> 16) & 0xFF)
    @inbounds out[pos]   = UInt8(e & 0xFF)
    @inbounds out[pos+1] = UInt8((e >> 8) & 0xFF)
    rb.bits  = _shl(rb.bits, nb_total)
    rb.nbits  -= nb_total
    return nb_total > nb1 ? 2 : 1
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
