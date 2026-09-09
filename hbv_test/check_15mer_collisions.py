import sys
sys.path.insert(0, '/Projects/ACCESS_IVFS')

from pathlib import Path
from genstore_targetcall import hybrid_filter as hf

BASE = Path('/Projects/ACCESS_IVFS/hbv_test')

# Check collision stats for seed=15, hash_bits=32
index = hf.load_index(BASE / 'seed_scan' / 'hbv.k15.h32.gsi')
hashes = index['hashes']
n_kmers = index['indexed_kmers']
n_unique = len(hashes)
max_occ = max(len(v) for v in hashes.values())

print(f"seed=15, hash_bits=32:")
print(f"  indexed k-mers: {n_kmers}")
print(f"  unique hashes:  {n_unique}")
print(f"  collisions:     {n_kmers - n_unique}")
print(f"  max occurrences per hash: {max_occ}")

# CAM capacity check (complementary 2 cell/bit, 512x512 = 262144 cell = 131072 bit)
cam_cells = 512 * 512
cam_bits_1cell = cam_cells          # 1T1R
cam_bits_2cell = cam_cells // 2     # complementary 2R

for bits in [32, 40, 48]:
    for label, eff_bits in [("1 cell/bit", cam_bits_1cell), ("2 cell/bit (互补)", cam_bits_2cell)]:
        slots = eff_bits // bits
        status = "OK" if slots >= n_kmers else "NOT ENOUGH"
        print(f"  {bits}-bit hash, {label}: {slots} slots vs {n_kmers} k-mers -> {status}")
