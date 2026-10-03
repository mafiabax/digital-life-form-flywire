import tempfile
import unittest
from pathlib import Path

from v262_pp3_path_reconstruction import parse_swc, reduce_pp3_tree


class V262Tests(unittest.TestCase):
    def test_root_is_not_double_counted_in_reduction(self):
        swc = """1 1 0 0 0 1 -1
2 5 1 0 0 1 1
3 0 2 0 0 1 2
4 0 3 0 0 1 2
5 0 0 1 0 1 1
"""
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "fixture.swc"
            path.write_text(swc, encoding="utf-8")
            tree = parse_swc(path)
            reduction = reduce_pp3_tree(tree, 1, {1, 2, 3, 4, 5})

            self.assertEqual(reduction["start_node_count_including_root"], 2)
            self.assertEqual(reduction["leaf_count"], 3)
            self.assertEqual(reduction["reduced_node_count"], 5)
            self.assertEqual(reduction["reduced_edge_count"], 4)
            self.assertAlmostEqual(reduction["reduced_cable_nm"], 5.0, places=9)

    def test_root_coordinate_is_preserved(self):
        swc = """1 1 10 20 30 1 -1
2 5 11 20 30 1 1
3 0 12 20 30 1 2
4 0 10 21 30 1 1
"""
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "fixture.swc"
            path.write_text(swc, encoding="utf-8")
            tree = parse_swc(path)
            reduction = reduce_pp3_tree(tree, 1, {1, 2, 3, 4})
            self.assertEqual(reduction["root_node_id"], 1)
            self.assertEqual(reduction["root_coordinate_nm"], (10.0, 20.0, 30.0))


if __name__ == "__main__":
    unittest.main()
