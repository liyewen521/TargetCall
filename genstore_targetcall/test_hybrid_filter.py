import tempfile
import unittest
from pathlib import Path

from genstore_targetcall.hybrid_filter import (
    SequenceRecord,
    build_index,
    genstore_hash,
    load_index,
    reverse_complement,
    score_read,
)


class HybridFilterTests(unittest.TestCase):
    def test_hash_is_canonical_across_strands(self):
        sequence = "ACGTTGCAACGTTGCAACGTTGCAACGTTGC"
        forward = genstore_hash(sequence, 48)
        reverse = genstore_hash(reverse_complement(sequence), 48)
        self.assertEqual(forward[0], reverse[0])
        self.assertNotEqual(forward[1], reverse[1])

    def test_hash_matches_genstore_minimap2_known_value(self):
        sequence = "ATTTTACTATTTTATTTAGTGTCTAGAAAAA"
        self.assertEqual(genstore_hash(sequence, 64), (4800009423800280863, False))

    def test_forward_and_reverse_reads_form_coherent_clusters(self):
        reference_sequence = "ACGT" * 20 + "GATTACAGCTTAGGCTAAGCTTACCGGATC" + "TGCA" * 20
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            reference = directory / "reference.fasta"
            index_path = directory / "reference.gsi"
            reference.write_text(f">target\n{reference_sequence}\n")
            build_index(reference, index_path, seed_length=15, hash_bits=48)
            index = load_index(index_path)
            forward = SequenceRecord("forward", reference_sequence[60:120])
            reverse = SequenceRecord("reverse", reverse_complement(forward.sequence))

            forward_score = score_read(forward, index, 15, 4, 1000)
            reverse_score = score_read(reverse, index, 15, 4, 1000)

            self.assertGreaterEqual(forward_score[2], 4)
            self.assertGreaterEqual(reverse_score[2], 4)
            self.assertTrue(forward_score[4])
            self.assertFalse(reverse_score[4])


if __name__ == "__main__":
    unittest.main()
