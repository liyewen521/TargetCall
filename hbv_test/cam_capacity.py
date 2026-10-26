import math

hbv_len = 3182
cells = 512 * 512

print('HBV reference length:', hbv_len, 'bp')
print('array cells: 512x512 =', cells, '(complementary 2 cell/bit -> effective', cells // 2, 'bit)')
print()
print('=== number of k-mers vs array capacity (total bits) ===')
for k in [15, 21, 31]:
    n = hbv_len - k + 1
    bits = 2 * k
    cell_comp = bits * 2  # complementary 2 cell/bit
    total_cells = n * cell_comp
    ratio = total_cells / cells
    print(f'  {k}-mer: {n} entries, {bits} bit/key (complementary {cell_comp} cell), '
          f'total {total_cells:,} cell, {ratio*100:.0f}% of array')

print()
print('=== standard CAM structure (one key per matchline) ===')
print('  512 matchline = 512 key slots (hard constraint)')
for k in [15, 21, 31]:
    n = hbv_len - k + 1
    print(f'  {k}-mer: {n} k-mers need {n} rows -> {n/512:.1f}x short, '
          f'need {math.ceil(n/512)} 512x512 arrays')
