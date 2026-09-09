import gzip, pickle, sys

with gzip.open('targetcall-reproduction/Monkeypox_virus.k31.gsi', 'rb') as f:
    p = pickle.load(f)
h = p['hashes']

one_key = next(iter(h))
one_list = h[one_key]
one_entry = one_list[0]

print('一个 hash 键 (int) 大小:', sys.getsizeof(one_key), 'bytes')
print('一个值 (list, 1项) 大小:', sys.getsizeof(one_list), 'bytes')
print('一项 (tuple) 大小:', sys.getsizeof(one_entry), 'bytes')
print('  tuple 内 contig_id int:', sys.getsizeof(one_entry[0]), 'bytes')
print('  tuple 内 position int:', sys.getsizeof(one_entry[1]), 'bytes')
print('  tuple 内 reverse bool:', sys.getsizeof(one_entry[2]), 'bytes')
print('hash 值 bit 宽:', one_key.bit_length(), 'bits')
print('position 最大值:', max(e[1] for v in h.values() for e in v))
