import math

hbv_len = 3182
cells = 512 * 512

print('HBV 参考长度:', hbv_len, 'bp')
print('阵列 cell 数: 512x512 =', cells, '(互补型 2 cell/bit -> 有效', cells // 2, 'bit)')
print()
print('=== k-mer 数 vs 阵列容量 (bit 总量口径) ===')
for k in [15, 21, 31]:
    n = hbv_len - k + 1
    bits = 2 * k
    cell_comp = bits * 2  # 互补型 2 cell/bit
    total_cells = n * cell_comp
    ratio = total_cells / cells
    print(f'  {k}-mer: {n} 个, {bits} bit/key (互补 {cell_comp} cell), '
          f'总 {total_cells:,} cell, 占阵列 {ratio*100:.0f}%')

print()
print('=== 标准 CAM 结构 (每 matchline 一个 key) ===')
print('  512 matchline = 512 个 key 槽位 (硬约束)')
for k in [15, 21, 31]:
    n = hbv_len - k + 1
    print(f'  {k}-mer: {n} 个 k-mer 需要 {n} 行 -> 差 {n/512:.1f} 倍, '
          f'需 {math.ceil(n/512)} 块 512x512 阵列')
