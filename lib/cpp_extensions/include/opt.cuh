#pragma once
#include "common.cuh"
#include "def.cuh"

// Map a logical element column to a physical shared-memory element column.
// Specialized for the current 16-bit, 16-byte-vector shared-memory layout.
DEVICE_CONSTEXPR index_t
SwizzledCol(index_t row, index_t col) {
    // Each vector contains 8 FP16 elements and occupies 16 bytes.
    constexpr index_t kSwizzleVectorElems = kElemsPerCpAsync;

    // The vector permutation repeats every 8 rows.
    constexpr index_t kSwizzleRowPeriod = kLdMatrixSize;

    // Each column region contains 8 vectors:
    // 64 FP16 elements, or 128 bytes.
    constexpr index_t kSwizzleVectorsPerRegion = kSwizzleRowPeriod;
    constexpr index_t kSwizzleRegionElems = kSwizzleVectorsPerRegion * kSwizzleVectorElems;

    // Guard the assumptions of this specialized layout.
    static_assert(kSwizzleVectorElems == 8);
    static_assert(kSwizzleRowPeriod == 8);
    static_assert(kHeadDim % kSwizzleRegionElems == 0);

    // Locate the column region.
    const index_t region_idx = col / kSwizzleRegionElems;
    const index_t col_in_region = col % kSwizzleRegionElems;

    // Locate the vector and preserve the element offset within it.
    const index_t vector_idx = col_in_region / kSwizzleVectorElems;
    const index_t elem_in_vector = col_in_region % kSwizzleVectorElems;

    // Permute vectors within the region according to the row.
    const index_t row_xor_mask = row % kSwizzleRowPeriod;
    const index_t swizzled_vector_idx = vector_idx ^ row_xor_mask;

    // Reconstruct the physical shared-memory column.
    const index_t region_start_col = region_idx * kSwizzleRegionElems;
    const index_t swizzled_col_in_region = swizzled_vector_idx * kSwizzleVectorElems + elem_in_vector;

    return region_start_col + swizzled_col_in_region;
}