import unittest
from coverage.preprocessing.coverage_functions import *
from config import *


class TestCoverage(unittest.TestCase):
    def test_open_gdb_file(self):
        self.coverage = Coverage()
        population_gdb = self.coverage.open_gdb_file(population_map)
        self.assertIsInstance(population_gdb, pd.DataFrame)
        self.assertTrue("population" in population_gdb.columns)
        self.assertTrue(population_gdb["population"].sum() > 0)

    def test_create_grided_map(self):
        self.fail()

    def test_pop_cell_creation(self):
        self.fail()


if __name__ == '__main__':
    unittest.main()
