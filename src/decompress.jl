# ============================================================
# Compressed block reading
#   Reference: RFC 8878 §3.1.1.3
# ============================================================



# ============================================================
# Literals reading
#   Reference: RFC 8878 §3.1.1.3
# ============================================================

# Read the literals section starting at data[pos].
# Returns (literals::Vector{UInt8}, bytes_consumed::Int).
function read_literals(data::Vector{UInt8}, pos::Int, state::DecompressState)
    br = ForwardBitReader(@view data[pos:end])
    litblock_type = read(br, 2)
    size_format = peek(br, 2)

    is_raw = litblock_type == 0
    is_rle = litblock_type == 1
    is_treeless = litblock_type == 3

    if is_raw || is_rle
        size_format_nbits = iseven(size_format) ? 1 :
                                                  2
        size_nbits, header_nbytes = iseven(size_format) ? ( 5, 1) :
                                    size_format == 1    ? (12, 2) :
                                                          (20, 3)
        skip(br, size_format_nbits)
        regen_size = Int(read(br, size_nbits))
        compressed_size = is_raw ? regen_size :
                                   1

        literals = state.literals_buf
        resize!(literals, regen_size + LITERALS_WILDCOPY_SLACK)

        block_start = pos + header_nbytes

        if is_raw
            copyto!(literals, 1, data, block_start, regen_size)
        else
            literals[1:regen_size] .= data[block_start]
        end

        return literals, header_nbytes + compressed_size
    else # Compressed (2) or Treeless (3)
        size_nbits, header_nbytes = size_format < 2  ? (10, 3) :
                                    size_format == 2 ? (14, 4) :
                                                       (18, 5)
        num_streams = size_format == 0 ? 1 :
                                         4
        
        skip(br, 2)
        regen_size = Int(read(br, size_nbits))
        compressed_size = Int(read(br, size_nbits))

        payload_start = pos + header_nbytes
        payload_end = payload_start + compressed_size - 1

        if is_treeless
            state.huffman !== nothing ||
                throw(ArgumentError("zstd: treeless literals but no prior Huffman table"))
            ht = state.huffman
            huf_start = payload_start
        else # Compressed
            ht, hdr_len = read_huffman_description((@view data[payload_start:payload_end]); scratch_buffers = (state.huf_weights, state.huf_rank_count, state.huf_rank_start))
            state.huffman = ht
            huf_start = payload_start + hdr_len
        end

        literals = state.literals_buf
        resize!(literals, regen_size + LITERALS_WILDCOPY_SLACK)

        if num_streams == 1
            stream_len = payload_end - huf_start + 1
            rb = ReverseBitReader(@view data[huf_start:huf_start + stream_len - 1])
            let p = 1
                while p ≤ regen_size
                    p += decode1x2_tail!(rb, ht, literals, p)
                end
            end
        else
            _decode_4streams!((@view data[huf_start:payload_end]), ht, literals, regen_size)
        end

        resize!(literals, regen_size + LITERALS_WILDCOPY_SLACK)  # trim dual-symbol slack

        return literals, header_nbytes + compressed_size
    end
end


# ============================================================
# Huffman tree loading
#   Reference: RFC 8878 §4.2.1
# ============================================================

function read_huffman_description(data::AbstractVector{UInt8}; scratch_buffers::Union{Nothing, Tuple{AbstractVector{UInt8}, AbstractVector{Int}, AbstractVector{Int}}} = nothing)
    headerByte = Int(data[1]) # RFC 8878 §4.2.1.1
    weights = scratch_buffers !== nothing ? scratch_buffers[1] : UInt8[]
    is_fse_encoded = headerByte < 128
    if is_fse_encoded
        nbytes = headerByte
        br = ForwardBitReader(@view data[2:nbytes + 1])
        _, table_log = _read_fse_weights!(weights, br, nbytes)
    else
        nsyms = headerByte - 127
        nbytes = (nsyms + 1) >> 1
        weightdata = @view data[2:nbytes + 1]
        _, table_log = _read_direct_weights!(weights, weightdata, nsyms)
    end
    scratch_buffers !== nothing && (scratch_buffers = scratch_buffers[2:3])
    ht = build_huffman_table!(weights, table_log; scratch_buffers)
    return ht, nbytes + 1
end

function _read_direct_weights!(weights::Vector{UInt8}, data::AbstractVector{UInt8}, nsyms::Int)
    nbytes = (nsyms + 1) >> 1
    resize!(weights, nsyms + 1)
    for i in 1:nbytes
        b = data[i]
        j = (i - 1) * 2 + 1
        weights[j] = (b >> 4) & 0x0f
        j + 1 ≤ nsyms &&
            (weights[j + 1] = b & 0x0f)
    end
    last_w, table_log = _infer_last_weight(weights)
    weights[end] = last_w
    return weights, table_log
end

# RFC 8878 §4.2.1.2
function _read_fse_weights!(weights::Vector{UInt8}, br::ForwardBitReader, byte_limit::Int)
    al, dist = read_fse_dist!(br, HUFTABLE_LOG_MAX)
    t = build_fse_table(dist, al)

    pos_after = byte_pos(br)
    n_remain = byte_limit - pos_after + 1
    n_remain > 0 ||
        throw(ArgumentError("zstd: no data for Huffman weight FSE stream"))

    rb = ReverseBitReader(@view br.data[pos_after:pos_after+n_remain-1])

    state1 = fse_init!(rb, t)
    state2 = fse_init!(rb, t)

    empty!(weights)
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

    br.pos = byte_limit + 1
    br.nbits = 0
    br.bits = UInt64(0)

    last_w, table_log = _infer_last_weight(weights)
    push!(weights, UInt8(last_w))

    return weights, table_log
end

function _infer_last_weight(weights::Vector{UInt8})
    total = Int(sum(w -> UInt64(1) << (Int(w) - 1), weights))
    total > 0 || return (1, 1)  # single symbol edge case
    table_log = _flog2(total) + 1
    p = UInt64(1) << table_log
    p > total || (table_log += 1; p <<= 1)
    rest = Int(p - total)
    rest > 0 && (rest & (rest - 1)) == 0 ||
        throw(ArgumentError("zstd: invalid Huffman weight sum"))
    last_w = _flog2(rest) + 1
    return last_w, table_log
end


# ============================================================
# Huffman stream decoding
#   Reference: RFC 8878 §4.2.2
# ============================================================

# Decode the four Huffman streams stored in `data` using the lookup table `ht` and store
# the result in `literals`. This code is tuned to promote LLVM SIMD instructions; changes
# in it or the functions it calls could break this. Use caution.
function _decode_4streams!(data::AbstractVector{UInt8}, ht::HuffmanTable{L},
                            literals::Vector{UInt8}, regen_size::Int) where L
    # Read stream-start indexes
    s1_start = 7
    s2_start = s1_start + Int(_le16(data, 1))
    s3_start = s2_start + Int(_le16(data, 3))
    s4_start = s3_start + Int(_le16(data, 5))
    s4_end = length(data)

    seg_n = (regen_size + 3) >> 2
    safe_n = 57 ÷ L
    oi = (1, 1 + seg_n, 1 + 2seg_n, 1 + 3seg_n)
    ends = (seg_n, 2seg_n, 3seg_n, regen_size)
    safeends = (ends[1] - 2safe_n, ends[2] - 2safe_n, ends[3] - 2safe_n, ends[4] - 2safe_n)

    # Phase 1: SIMD parallel processing of the four streams until at least one is exhausted (within safe window)
    rb4x = ReverseBitReaderX(
        @view(data[s1_start:s2_start-1]),
        @view(data[s2_start:s3_start-1]),
        @view(data[s3_start:s4_start-1]),
        @view(data[s4_start:s4_end]),
    )
    oi_vec = Vec{4, Int}(oi)
    safeends_vec = Vec{4, Int}(safeends)
    while all(oi_vec ≤ safeends_vec)
        refill_unchecked!(rb4x)
        for _ in 1:safe_n
            nread = decode4x2!(rb4x, ht, literals, oi_vec)
            oi_vec += nread
        end
    end
    oi = Tuple(oi_vec)  # spill Vec back to scalar for remaining phases

    # Phase 2A: SIMD parallel processing of the top pair of streams with the most work remaining
    r = (safeends[1] - oi[1], safeends[2] - oi[2],
         safeends[3] - oi[3], safeends[4] - oi[4])
    ia, ib, ic, id = sortperm([r[1], r[2], r[3], r[4]], rev=true)

    s1 = _extract_stream(rb4x, Val(1))
    s2 = _extract_stream(rb4x, Val(2))
    s3 = _extract_stream(rb4x, Val(3))
    s4 = _extract_stream(rb4x, Val(4))
    sv = (s1, s2, s3, s4)

    rbA = ReverseBitReaderX(sv[ia], sv[ib])
    oi_A = Vec{2, Int}((oi[ia], oi[ib]))
    se_A = Vec{2, Int}((safeends[ia], safeends[ib]))
    while all(oi_A ≤ se_A)
        refill_unchecked!(rbA)
        for _ in 1:safe_n
            nread = decode2x2!(rbA, ht, literals, oi_A)
            oi_A += nread
        end
    end
    ra_ia = _extract_stream(rbA, Val(1))
    ra_ib = _extract_stream(rbA, Val(2))

    # Phase 2B: SIMD parallel processing of the survivor with the last unexhausted stream
    ia_alive = oi_A[1] ≤ se_A[1]
    re_2a  = ia_alive ? ra_ia        : ra_ib
    oi_2a  = ia_alive ? Int(oi_A[1]) : Int(oi_A[2])
    se_2a  = ia_alive ? safeends[ia] : safeends[ib]

    rbB = ReverseBitReaderX(re_2a, sv[ic])
    oi_B = Vec{2, Int}((oi_2a, oi[ic]))
    se_B = Vec{2, Int}((se_2a, safeends[ic]))
    while all(oi_B ≤ se_B)
        refill_unchecked!(rbB)
        for _ in 1:safe_n
            nread = decode2x2!(rbB, ht, literals, oi_B)
            oi_B += nread
        end
    end
    rb_2b  = _extract_stream(rbB, Val(1))   # survivor-of-2a reader, updated
    rb_ic2 = _extract_stream(rbB, Val(2))   # ic reader, updated

    # Phase 2C: Process remaining unexhausted stream
    ie_alive = oi_B[1] ≤ se_B[1]
    re_2b  = ie_alive ? rb_2b  : rb_ic2
    oi_2b  = ie_alive ? Int(oi_B[1]) : Int(oi_B[2])
    se_2b  = ie_alive ? se_2a  : safeends[ic]
    while oi_2b ≤ se_2b
        refill!(re_2b)
        for _ in 1:safe_n
            nread = decode1x2!(re_2b, ht, literals, oi_2b)
            oi_2b += nread
        end
    end

    # Phase 3: Process any remaining tails of all four streams
    rbs = (ia_alive ? rb_2b  : ra_ia, ia_alive ? ra_ib  : rb_2b, rb_ic2, sv[id])
    oi = (
        ia_alive ? (ie_alive ? oi_2b : Int(oi_B[1])) : Int(oi_A[1]),
        ia_alive ? Int(oi_A[2]) : (ie_alive ? oi_2b : Int(oi_B[1])),
        ie_alive ? Int(oi_B[2]) : oi_2b,
        oi[id]
    )
    perm = (ia, ib, ic, id)

    let p = oi[1]; while p ≤ ends[perm[1]]; p += decode1x2_tail!(rbs[1], ht, literals, p); end; end
    let p = oi[2]; while p ≤ ends[perm[2]]; p += decode1x2_tail!(rbs[2], ht, literals, p); end; end
    let p = oi[3]; while p ≤ ends[perm[3]]; p += decode1x2_tail!(rbs[3], ht, literals, p); end; end
    let p = oi[4]; while p ≤ ends[perm[4]]; p += decode1x2_tail!(rbs[4], ht, literals, p); end; end

    return
end

# Read 1-2 symbols from 4 streams and return the number of symbols read
# Always writes 2 symbols even if only the first is valid; up to the caller to provide room
@inline function decode4x2!(rb::ReverseBitReaderX{4}, ht::HuffmanTable{L}, out::Vector{UInt8}, oi::Vec{4, Int}) where L
    i = peek(rb, Val(L))
    @inbounds entry = (
        ht[i[1] % Int + 1],
        ht[i[2] % Int + 1],
        ht[i[3] % Int + 1],
        ht[i[4] % Int + 1]
    )
    GC.@preserve out begin
        unsafe_store!.(Ptr{NTuple{2, UInt8}}.(pointer.(Ref(out), Tuple(oi))), htol.(getfield.(entry, :symbols)))
    end
    nbits_consumed = (Int(entry[1].stream_nbits), Int(entry[2].stream_nbits),
                      Int(entry[3].stream_nbits), Int(entry[4].stream_nbits))
    skip(rb, nbits_consumed)
    return Vec{4, Int64}((Int64(entry[1].nsymbols), Int64(entry[2].nsymbols),
                          Int64(entry[3].nsymbols), Int64(entry[4].nsymbols)))
end

# Read 1-2 symbols from 2 streams and return the number of symbols read
# Always writes 2 symbols even if only the first is valid; up to the caller to provide room
@inline function decode2x2!(rb::ReverseBitReaderX{2}, ht::HuffmanTable{L}, out::Vector{UInt8}, oi::Vec{2, Int}) where L
    i = peek(rb, Val(L))
    @inbounds entry = (
        ht[i[1] % Int + 1],
        ht[i[2] % Int + 1]
    )
    GC.@preserve out begin
        unsafe_store!.(Ptr{NTuple{2, UInt8}}.(pointer.(Ref(out), Tuple(oi))), htol.(getfield.(entry, :symbols)))
    end
    nbits_consumed = (Int(entry[1].stream_nbits), Int(entry[2].stream_nbits))
    skip(rb, nbits_consumed)
    return Vec{2, Int64}((Int64(entry[1].nsymbols), Int64(entry[2].nsymbols)))
end

# Read 1-2 symbols from 1 stream and return the number of symbols read
# Always writes 2 symbols even if only the first is valid; up to the caller to provide room
@inline function decode1x2!(rb::ReverseBitReader, ht::HuffmanTable{L}, out::Vector{UInt8}, o::Int) where L
    i = peek(rb, Val(L))
    @inbounds entry = ht[i[1] % Int + 1]
    GC.@preserve out begin
        unsafe_store!(Ptr{NTuple{2, UInt8}}(pointer(out, o)), htol(entry.symbols))
    end
    nbits_consumed = Int(entry.stream_nbits)
    skip(rb, nbits_consumed)
    return Int64(entry.nsymbols)
end

# Read 1-2 symbols from 1 stream and return the number of symbols read
# Does not write second symbol if it is not present
@inline function decode1x2_tail!(rb::ReverseBitReader, ht::HuffmanTable{L}, out::Vector{UInt8}, o::Int) where L
    rb.nbits ≥ L || refill!(rb)
    i = peek(rb, Val(L))
    entry = @inbounds ht.decode_table[i + 1]
    nbits_consumed = Int(entry.stream_nbits)
    @inbounds out[o] = entry.symbols[1]
    skip(rb, nbits_consumed)
    if entry.nsymbols == 2
        @inbounds out[o + 1] = entry.symbols[2]
    end
    return entry.nsymbols
end
