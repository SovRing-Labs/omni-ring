"""Safe MEM-1B smoke and evidence checks; never allocate the billion-row table."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / 'experiments' / 'mem1b'

class Mem1bTests(unittest.TestCase):
    def run_harness(self, out, *args):
        return subprocess.run([sys.executable, str(EXP / 'mem1b.py'), str(out), *map(str, args)],
                              cwd=ROOT, capture_output=True, text=True, timeout=60)

    def test_small_control_and_lookup(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'small.json'
            p = self.run_harness(out, 1000, 20, 8, 3, 5, 7, 13, 17)
            self.assertEqual(p.returncode, 0, p.stderr)
            r = json.loads(out.read_text())
            for key, n, lookups, width in [('MEM0_control_1365', 1365, 200, 64), ('MEM1B_1000', 1000, 20, 8)]:
                a = r[key]
                self.assertEqual((a['recall'], a['wrong_examples']), (1.0, []))
                self.assertEqual((a['N_stored'], a['lookups'], a['table_bytes']), (n, lookups, n * width))
                self.assertEqual(a['bytes_touched'], 45 * 5120 + width)
            before = out.read_bytes()
            self.assertNotEqual(self.run_harness(out, 1000).returncode, 0)
            self.assertEqual(out.read_bytes(), before)

    def test_invalid_arguments_before_allocation(self):
        cases = [(0,), (1, 0), (1, 1, -1), (1, 1, 8, 3), (1, 1, 8, 3, 3),
                 (1, 1, 8, 1, 5), (100, 1, 8, 3, 5), (1000000000,),
                 (1, 1, 8, 10007, 3), (1, 1, 8, 2147483647, 2147483629, 2147483587, 3)]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'bad.json'
            for args in cases:
                with self.subTest(args=args):
                    p = self.run_harness(out, *args)
                    self.assertNotEqual(p.returncode, 0)
                    self.assertFalse(out.exists())

    def test_historical_evidence_and_hashes(self):
        provenance = json.loads((EXP / 'PROVENANCE.json').read_text())
        for path, sha in provenance['published_snapshot_sha256'].items():
            self.assertEqual(hashlib.sha256((EXP / path).read_bytes()).hexdigest(), sha)
        r = json.loads((EXP / 'results/xeon-8481c.json').read_text())
        a = r['MEM1B_1000000000']
        self.assertEqual((a['N_stored'], a['lookups'], a['recall']), (1000000000, 1000, 1.0))
        self.assertEqual(a['table_bytes'], 8000000000)
        self.assertEqual(a['bytes_touched'], sum(a['moduli']) * a['dimension'] + a['payload_bytes'])
        self.assertEqual(r['MEM0_control_1365']['recall'], 1.0)

if __name__ == '__main__':
    unittest.main()
