# Finite State Entropy (FSE) decode table
#   Reference: RFC 8878 §4.1
# Mutable so the hot path (build_fse_table!, decompress.jl) can update
# `accuracy_log` on a persistent, slot-owned instance instead of allocating a
# fresh wrapper every FSE_Compressed-mode block — the backing array itself was
# already reused via resize! below.
#
# One packed 8-byte entry per state, the same layout as libzstd's
# ZSTD_seqSymbol:
#
#   bits  0..15  next_state  — state baseline, before adding the transition bits
#   bits 16..23  nb_bits     — width of this state's transition read
#   bits 24..31  nb_add      — extra value bits to read for this symbol
#   bits 32..63  base_value  — value baseline for this symbol
#
# The last two are the point. Decoding a sequence needs a symbol's baseline and
# extra-bit count, and those used to be a second lookup indexed by the symbol
# the FSE table had just produced -- a dependent
# entry -> code -> LITERALS_LENGTH_BASELINE[code] hop, two load latencies deep,
# feeding `total_n` which gates everything else in the sequence. Baking them in
# at table-build time makes a state lookup a single load.
#
# Consequently the symbol itself is no longer stored: for sequence tables it
# was only ever a stepping stone to those two values, and for the Huffman
# weight table (which has no baselines) it is stored in `base_value`.
mutable struct FSEDistTable
    accuracy_log::Int
    entries::Vector{UInt64}
end

@inline _fse_pack(next_state::UInt16, nb_bits::UInt8, nb_add::UInt8, base_value::UInt32) =
    UInt64(next_state) | (UInt64(nb_bits) << 16) | (UInt64(nb_add) << 24) |
    (UInt64(base_value) << 32)

@inline _fse_next(e::UInt64) = Int(e & 0xffff)
@inline _fse_nb(e::UInt64)   = Int((e >> 16) & 0xff)
@inline _fse_add(e::UInt64)  = Int((e >> 24) & 0xff)
@inline _fse_base(e::UInt64) = Int(e >> 32)

# Run-length encoding table: a single implicit state, so it just carries the
# one packed entry (next_state and nb_bits zero, since it never transitions).
struct RLEDistTable
    entry::UInt64
end

# What a table's symbols mean, so the builder can bake each symbol's value
# baseline and extra-bit count into its entry. `_sym_max` is the largest symbol
# the mapping is defined for; checking it here is what lets the sequence hot
# path index nothing but the table itself.
struct SeqLL end
struct SeqML end
struct SeqOF end
struct RawSym end   # Huffman weights: the "value" is the symbol itself

@inline _sym_max(::SeqLL)  = MAX_LITERALS_LENGTH
@inline _sym_max(::SeqML)  = MAX_MATCH_LENGTH
@inline _sym_max(::SeqOF)  = MAX_OFFSET_CODE
@inline _sym_max(::RawSym) = 255

@inline _sym_value(::SeqLL, s::Int) =
    (@inbounds(LITERALS_LENGTH_BASELINE[s + 1]), @inbounds(LITERALS_LENGTH_EXTRA_BITS[s + 1]))
@inline _sym_value(::SeqML, s::Int) =
    (@inbounds(MATCH_LENGTH_BASELINE[s + 1]), @inbounds(MATCH_LENGTH_EXTRA_BITS[s + 1]))
# Offset codes are their own extra-bit count, and their baseline is 1 << code.
@inline _sym_value(::SeqOF, s::Int) = (UInt32(1) << s, UInt8(s))
@inline _sym_value(::RawSym, s::Int) = (UInt32(s), UInt8(0))

@noinline _throw_fse_symbol(s, m) =
    throw(ArgumentError("zstd: FSE symbol $s exceeds maximum $m for this table"))

@inline _rle_table(kind, sym::Int) =
    (sym ≤ _sym_max(kind) || _throw_fse_symbol(sym, _sym_max(kind));
     RLEDistTable(_fse_pack(UInt16(0), UInt8(0), _sym_value(kind, sym)[2],
                            _sym_value(kind, sym)[1])))

# Fill pre-allocated backing arrays for an FSE decode table from a normalized
# probability distribution. norm[i+1] is the probability of symbol i; -1
# means "probability 1/tableSize". Shared by the raw-array hot path below and
# the slot-owning `build_fse_table!` (decompress.jl, defined after
# FSEDistTableSlot).
function _fill_fse_table!(norm::AbstractVector{<:Integer}, accuracy_log::Int,
                           entries::Vector{UInt64}, occ::Vector{Int}, kind)
    table_size = 1 << accuracy_log
    resize!(entries, table_size)
    resize!(occ, length(norm))
    fill!(occ, 0)

    # Every symbol must be one the kind's value mapping is defined for. The
    # callers all bound `length(norm)` already, but checking here is what makes
    # the `@inbounds` in `_sym_value` safe on its own terms.
    length(norm) ≤ _sym_max(kind) + 1 ||
        _throw_fse_symbol(length(norm) - 1, _sym_max(kind))

    # --- Spread: place -1 symbols at the end, others via step pattern ---
    # The spread pass parks each state's symbol in the low bits; the second pass
    # reads it back and overwrites the slot with the finished entry. Every state
    # is covered (the counts sum to table_size), so none is left half-built.
    high = table_size - 1
    for i in eachindex(norm)
        norm[i] == -1 || continue
        entries[high + 1] = UInt64(UInt8(i - 1))
        high -= 1
    end

    step = (table_size >> 1) + (table_size >> 3) + 3
    mask = table_size - 1
    pos  = 0
    for (i, c) in enumerate(norm)
        for _ in 1:c
            entries[pos + 1] = UInt64(UInt8(i - 1))
            pos = (pos + step) & mask
            while pos > high
                pos = (pos + step) & mask
            end
        end
    end

    # --- Build per-state decode entries ---
    for i in 1:table_size
        s = Int(entries[i] & 0xff)
        c = norm[s + 1]
        c == -1 && (c = 1)
        j = occ[s + 1]
        occ[s + 1] += 1
        ci = c + j
        n = accuracy_log - _flog2(ci)
        base_value, nb_add = _sym_value(kind, s)
        entries[i] = _fse_pack(UInt16((ci << n) - table_size), UInt8(n), nb_add, base_value)
    end
    return nothing
end

# Hot-path: caller supplies pre-allocated backing arrays; they are resized
# and filled in-place, but this still allocates a fresh `FSEDistTable`
# wrapper each call. Kept for the cold path below; the truly hot call site
# (decompress.jl's read_distribution_table!) uses `build_fse_table!` instead,
# which reuses the wrapper too.
function build_fse_table(norm::AbstractVector{<:Integer}, accuracy_log::Int,
                          entries::Vector{UInt64}, occ::Vector{Int}, kind)
    _fill_fse_table!(norm, accuracy_log, entries, occ, kind)
    return FSEDistTable(accuracy_log, entries)
end

# Cold-path: allocates its own backing arrays (used by __init__, parse_dictionary, etc.)
function build_fse_table(norm::AbstractVector{<:Integer}, accuracy_log::Int, kind)
    table_size = 1 << accuracy_log
    return build_fse_table(norm, accuracy_log,
                           Vector{UInt64}(undef, table_size),
                           Vector{Int}(undef, length(norm)), kind)
end

# Read an FSE normalized distribution from the forward bitstream.
# Returns (accuracy_log, norm_counts).
# Implements the reference zstd sliding-threshold algorithm.
# Hot-path: caller supplies a reusable norm buffer (emptied on entry).
function read_fse_dist!(br::ForwardBitReader, norm::Vector{Int16})
    accuracy_log = Int(read(br, 4)) + 5
    table_size   = 1 << accuracy_log

    empty!(norm)
    remaining = table_size + 1   # reference zstd initialises to tableSize+1
    threshold = table_size
    nbits     = accuracy_log + 1

    while remaining > 1
        br.nbits < nbits && refill!(br)

        # max = number of values encodable in the short (nbits-1 bit) path
        max_val = (2 * threshold - 1) - remaining
        low     = Int(br.bits & UInt64(threshold - 1))   # peek lower nbits-1 bits

        local count::Int
        if low < max_val
            # Short path: value from nbits-1 bits
            count = low
            br.bits  >>>= (nbits - 1)
            br.nbits  -= (nbits - 1)
        else
            # Long path: value from nbits bits
            count = Int(br.bits & UInt64(2 * threshold - 1))
            if count ≥ threshold
                count -= max_val
            end
            br.bits  >>>= nbits
            br.nbits  -= nbits
        end

        count -= 1   # "extra accuracy" offset: value 0 → count -1 (low-prob)

        push!(norm, Int16(count))

        if count == 0
            # Zero-run: chained 2-bit repeat count (RFC 8878 §4.1.1)
            while true
                br.nbits < 2 && refill!(br)
                r = Int(br.bits & 3)
                br.bits  >>>= 2
                br.nbits  -= 2
                for _ in 1:r
                    push!(norm, Int16(0))
                end
                r < 3 && break
            end
        else
            remaining -= count < 0 ? -count : count
        end

        # Shrink threshold/nbits as remaining decreases.
        while remaining < threshold
            nbits     -= 1
            threshold >>= 1
        end
    end

    align_to_byte!(br)
    return accuracy_log, norm
end

# Cold-path: allocates its own norm buffer (used by parse_dictionary, _decode_fse_weights, etc.)
function read_fse_dist!(br::ForwardBitReader, max_sym::Int)
    norm = Int16[]
    sizehint!(norm, max_sym + 1)
    return read_fse_dist!(br, norm)
end

# ------- Dist Table state machine helpers -------

@inline dist_table_init!(rb::ReverseBitReader, t::FSEDistTable) =
    Int(read(rb, t.accuracy_log))

# Fetch the whole packed entry for a state. Deliberately bounds-checked: the
# state comes from bitstream data, and this one check is what makes every field
# read below safe, since those are pure bit extraction on an already-loaded
# value.
@inline dist_table_entry(t::FSEDistTable, state::Int) = t.entries[state + 1]

# RLE tables have table log 0, so state init consumes 0 bits (the reference
# decoder's FSE_initDState reads tableLog bits) and the state is always 0.
@inline dist_table_init!(::ReverseBitReader, ::RLEDistTable) = 0
@inline dist_table_entry(t::RLEDistTable, ::Int) = t.entry

# For the Huffman weight table (built with `RawSym`) the symbol is what got
# baked into `base_value`.
@inline dist_table_peek(t::FSEDistTable, state::Int) = _fse_base(dist_table_entry(t, state))

# Update without checking for underflow (allows overflow detection after).
@inline function _fse_update_unchecked(rb::ReverseBitReader, t::FSEDistTable, state::Int)
    e    = dist_table_entry(t, state)
    bits = Int(_read_bits_unchecked!(rb, _fse_nb(e)))
    return _fse_next(e) + bits
end
