#include "common.cuh"
#include "def.cuh"

template <int kBanksPerVec4Access = 8, int kElemsPerBank = 8, typename index_t = int>
DEVICE_CONSTEXPR index_t
SwizzledCol(const index_t &row, const index_t &col) {
    // Restrict the swizzled column to the
    // (8, 128) byte region it's in.
    // Not strictly necessary, but we'll need it in later kernels.
    const index_t region_row = row % kBanksPerVec4Access;
 
    // Convert column byte offset to 16B bank index since we have 8 banks of 16B each.
    // This transforms the column coordinate from element space to bank space
    const index_t bank_col = col / kElemsPerBank;
    
    // Preserve the byte offset within each 16B bank for non-vectorized RF→SMEM stores
    // This ensures threads in the same 4-thread group maintain their relative positions
    const index_t bank_offset = col % kElemsPerBank;
 
    // Apply XOR swizzling to distribute consecutive row accesses across different banks
    // Then reconstruct the final column address by scaling back to element space
    return ((region_row ^ bank_col) * kElemsPerBank) + bank_offset;
}