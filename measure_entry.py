import gzip, pickle, sys

with gzip.open('targetcall-reproduction/Monkeypox_virus.k31.gsi', 'rb') as f:
    p = pickle.load(f)
h = p['hashes']

one_key = next(iter(h))
one_list = h[one_key]
one_entry = one_list[0]

print('size of one hash key (int):', sys.getsizeof(one_key), 'bytes')
print('size of one value (list, 1 item):', sys.getsizeof(one_list), 'bytes')
print('size of one entry (tuple):', sys.getsizeof(one_entry), 'bytes')
print('  contig_id int inside tuple:', sys.getsizeof(one_entry[0]), 'bytes')
print('  position int inside tuple:', sys.getsizeof(one_entry[1]), 'bytes')
print('  reverse bool inside tuple:', sys.getsizeof(one_entry[2]), 'bytes')
print('hash value bit width:', one_key.bit_length(), 'bits')
print('max position:', max(e[1] for v in h.values() for e in v))
