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
