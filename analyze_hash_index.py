import gzip, pickle, sys, os

path = '/Projects/ACCESS_IVFS/targetcall-reproduction/Monkeypox_virus.k31.gsi'

# File size on disk
raw_size = os.path.getsize(path)
print(f"index file (gzip compressed): {raw_size:,} bytes = {raw_size/1024/1024:.2f} MB")

with gzip.open(path, 'rb') as f:
    payload = pickle.load(f)

print(f"\n=== index metadata ===")
print(f"version:        {payload['version']}")
print(f"seed_length:    {payload['seed_length']} bp")
print(f"hash_bits:      {payload['hash_bits']} bits")
print(f"indexed_kmers:  {payload['indexed_kmers']:,}")
print(f"contigs:        {payload['contigs']}")

hashes = payload['hashes']
print(f"\n=== hash table structure ===")
print(f"unique hash keys: {len(hashes):,}")
print(f"total entries (k-mers): {sum(len(v) for v in hashes.values()):,}")

# Each value is List[Tuple[contig_id, position, reverse]]
# Measure per-entry size in Python memory (approx)
# Analyze distribution of entries per key
from collections import Counter
entry_counts = Counter(len(v) for v in hashes.values())
print(f"\n=== positions-per-key distribution ===")
for count, nkeys in sorted(entry_counts.items()):
    print(f"  {count} positions: {nkeys:,} keys")

# Average
avg = sum(len(v) for v in hashes.values()) / len(hashes)
print(f"\naverage {avg:.3f} positions per key")

# Estimate in-memory size of the dict
# Each key: Python int (48-bit -> small int object ~28 bytes)
# Each entry tuple (contig_id int, position int, reverse bool) -> ~64-72 bytes
# Each list overhead
total_entries = sum(len(v) for v in hashes.values())
print(f"\n=== memory estimate (uncompressed) ===")
print(f"keys (int):    {len(hashes):,} × ~28B = {len(hashes)*28/1024/1024:.2f} MB")
print(f"list overhead:    {len(hashes):,} × ~56B = {len(hashes)*56/1024/1024:.2f} MB")
print(f"entries (tuple):  {total_entries:,} × ~72B = {total_entries*72/1024/1024:.2f} MB")
approx = (len(hashes)*28 + len(hashes)*56 + total_entries*72) / 1024 / 1024
print(f"rough total:    ~{approx:.2f} MB")
