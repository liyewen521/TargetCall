import gzip, pickle, sys, os

path = '/Projects/ACCESS_IVFS/targetcall-reproduction/Monkeypox_virus.k31.gsi'

# File size on disk
raw_size = os.path.getsize(path)
print(f"索引文件 (gzip 压缩): {raw_size:,} bytes = {raw_size/1024/1024:.2f} MB")

with gzip.open(path, 'rb') as f:
    payload = pickle.load(f)

print(f"\n=== 索引元数据 ===")
print(f"version:        {payload['version']}")
print(f"seed_length:    {payload['seed_length']} bp")
print(f"hash_bits:      {payload['hash_bits']} bits")
print(f"indexed_kmers:  {payload['indexed_kmers']:,}")
print(f"contigs:        {payload['contigs']}")

hashes = payload['hashes']
print(f"\n=== 哈希表结构 ===")
print(f"唯一 hash 键数: {len(hashes):,}")
print(f"总项数 (k-mer): {sum(len(v) for v in hashes.values()):,}")

# Each value is List[Tuple[contig_id, position, reverse]]
# Measure per-entry size in Python memory (approx)
# Analyze distribution of entries per key
from collections import Counter
entry_counts = Counter(len(v) for v in hashes.values())
print(f"\n=== 每个键的位置数分布 ===")
for count, nkeys in sorted(entry_counts.items()):
    print(f"  {count} 个位置: {nkeys:,} 个键")

# Average
avg = sum(len(v) for v in hashes.values()) / len(hashes)
print(f"\n平均每个键 {avg:.3f} 个位置")

# Estimate in-memory size of the dict
# Each key: Python int (48-bit -> small int object ~28 bytes)
# Each entry tuple (contig_id int, position int, reverse bool) -> ~64-72 bytes
# Each list overhead
total_entries = sum(len(v) for v in hashes.values())
print(f"\n=== 内存估算 (未压缩) ===")
print(f"键 (int):    {len(hashes):,} × ~28B = {len(hashes)*28/1024/1024:.2f} MB")
print(f"列表开销:    {len(hashes):,} × ~56B = {len(hashes)*56/1024/1024:.2f} MB")
print(f"项 (tuple):  {total_entries:,} × ~72B = {total_entries*72/1024/1024:.2f} MB")
approx = (len(hashes)*28 + len(hashes)*56 + total_entries*72) / 1024 / 1024
print(f"粗略合计:    ~{approx:.2f} MB")
