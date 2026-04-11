# ============================================================
# FSE decode table
#   Reference: RFC 8878 §4.1
# ============================================================

struct FSETable
    accuracy_log::Int
    symbols  ::Vector{UInt8}
    nb_bits  ::Vector{UInt8}
    baselines::Vector{UInt32}
end

# Trivial FSE table for RLE mode: every state emits the same symbol with
# zero extra bits.  Stored as a single byte — no heap allocation.
struct RLEFSETable
    symbol::UInt8
end

# floor(log2(n)) for n ≥ 1
@inline _flog2(n::Int) = 63 - leading_zeros(UInt64(n))

# Build an FSE decode table from a normalized probability distribution.
# norm[i+1] is the probability of symbol i; -1 means "probability 1/tableSize".
# Hot-path: caller supplies pre-allocated backing arrays; they are resized and filled in-place.
function build_fse_table(norm::AbstractVector{<:Integer}, accuracy_log::Int,
                          syms::Vector{UInt8}, nb::Vector{UInt8},
                          base::Vector{UInt32}, occ::Vector{Int})
    table_size = 1 << accuracy_log
    resize!(syms, table_size)
    resize!(nb,   table_size)
    resize!(base, table_size)
    resize!(occ,  length(norm))
    fill!(occ, 0)

    # --- Spread: place -1 symbols at the end, others via step pattern ---
    high = table_size - 1
    for s in 0:length(norm)-1
        norm[s+1] == -1 || continue
        syms[high+1] = UInt8(s)
        high -= 1
    end

    step = (table_size >> 1) + (table_size >> 3) + 3
    mask = table_size - 1
    pos  = 0
    for s in 0:length(norm)-1
        c = Int(norm[s+1])
        c > 0 || continue
        for _ in 1:c
            syms[pos+1] = UInt8(s)
            pos = (pos + step) & mask
            while pos > high
                pos = (pos + step) & mask
            end
        end
    end

    # --- Build per-state decode entries ---
    for x in 0:table_size-1
        s  = Int(syms[x+1])
        c  = Int(norm[s+1]);  c == -1 && (c = 1)
        i  = occ[s+1];        occ[s+1] += 1
        ci = c + i
        n  = accuracy_log - _flog2(ci)
        nb[x+1]   = UInt8(n)
        base[x+1] = UInt32((ci << n) - table_size)
    end

    return FSETable(accuracy_log, syms, nb, base)
end

# Cold-path: allocates its own backing arrays (used by __init__, parse_dictionary, etc.)
function build_fse_table(norm::AbstractVector{<:Integer}, accuracy_log::Int)
    table_size = 1 << accuracy_log
    return build_fse_table(norm, accuracy_log,
                           Vector{UInt8}(undef, table_size),
                           Vector{UInt8}(undef, table_size),
                           Vector{UInt32}(undef, table_size),
                           Vector{Int}(undef, length(norm)))
end

# Read an FSE normalized distribution from the forward bitstream.
# Returns (accuracy_log, norm_counts).
# Implements the exact reference zstd sliding-threshold algorithm.
# Hot-path: caller supplies a reusable norm buffer (emptied on entry).
function read_fse_dist!(br::ForwardBitReader, max_sym::Int, norm::Vector{Int16})
    accuracy_log = Int(read_bits!(br, 4)) + 5
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
    return read_fse_dist!(br, max_sym, norm)
end

# ------- FSE state machine helpers -------

@inline fse_init!(rb::ReverseBitReader, t::FSETable) =
    Int(read_bits!(rb, t.accuracy_log))

@inline fse_peek(t::FSETable, state::Int) = Int(t.symbols[state+1])

@inline function fse_update!(rb::ReverseBitReader, t::FSETable, state::Int)
    nb   = Int(t.nb_bits[state+1])
    bits = Int(read_bits!(rb, nb))
    return Int(t.baselines[state+1]) + bits
end

# RLE variants — accuracy_log is always 1, both states emit the same symbol,
# and transitions consume 0 bits.
@inline fse_init!(rb::ReverseBitReader, t::RLEFSETable) =
    (read_bits!(rb, 1); 0)   # consume the 1-bit state init, state is always 0

@inline fse_peek(t::RLEFSETable, state::Int) = Int(t.symbol)

@inline fse_update!(rb::ReverseBitReader, t::RLEFSETable, state::Int) = 0

# Helpers for the batched sequence reader: retrieve the transition width and
# baseline for the current state without consuming any bits.
@inline _fse_nb_bits(t::FSETable,    state::Int) = Int(@inbounds t.nb_bits[state + 1])
@inline _fse_nb_bits(t::RLEFSETable, state::Int) = 0

@inline _fse_baseline(t::FSETable,    state::Int) = Int(@inbounds t.baselines[state + 1])
@inline _fse_baseline(t::RLEFSETable, state::Int) = 0

# Update without checking for underflow (allows overflow detection after).
@inline function _fse_update_unchecked(rb::ReverseBitReader, t::FSETable, state::Int)
    nb   = Int(t.nb_bits[state+1])
    bits = Int(_read_bits_unchecked!(rb, nb))
    return Int(t.baselines[state+1]) + bits
end

# Read an FSE table according to the given mode byte bits.
# mode: 0=predefined, 1=RLE, 2=FSE_Compressed, 3=Repeat
# Returns the FSETable and advances br.
function read_fse_table!(br::ForwardBitReader, default::FSETable,
                         prev::Union{FSETable,RLEFSETable,Nothing},
                         mode::Int, max_sym::Int, max_al::Int,
                         syms::Vector{UInt8}, nb::Vector{UInt8},
                         base::Vector{UInt32}, occ::Vector{Int},
                         norm::Vector{Int16})
    if mode == 0
        return default
    elseif mode == 1
        sym = read_bits!(br, 8)   # RLE: single symbol
        return RLEFSETable(UInt8(sym))
    elseif mode == 2
        al, dist = read_fse_dist!(br, max_sym, norm)
        al ≤ max_al || throw(ArgumentError("zstd: accuracy log $al exceeds maximum $max_al"))
        length(dist) ≤ max_sym + 1 || throw(ArgumentError("zstd: FSE distribution has $(length(dist)) symbols, maximum is $(max_sym + 1)"))
        return build_fse_table(dist, al, syms, nb, base, occ)
    else   # mode == 3: repeat previous table (RFC 8878 §3.1.1.3.3.2)
        prev !== nothing || throw(ArgumentError("zstd: repeat mode but no previous FSE table"))
        return prev
    end
end
