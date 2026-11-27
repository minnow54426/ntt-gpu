"""Exercise real Git handoffs with small synthetic inputs; never run GPU commands."""
import contextlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
import prepare_rental_bundle as handoff


class HandoffTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.repo = self.base / 'source'
        scripts = self.repo / 'scripts'
        scripts.mkdir(parents=True)
        for name in ('prepare_rental_bundle.py', 'gpu_rental_batch.py', 'gpu_session.py',
                     'benchmark_batch.py', 'benchmark_report.py', 'analyze_gpu_batch.py'):
            shutil.copyfile(ROOT / 'scripts' / name, scripts / name)
        (self.repo / '.gitignore').write_text('__pycache__/\nbuild/\nbench-results-*/\n')
        self.git('init', '-q')
        self.git('add', '.')
        self.commit('Fixture source')
        self.inputs = self.base / 'inputs'
        self.inputs.mkdir()
        for filename, suffix in (('vectors.json', '.txt'), ('conversion-references.json', '.conversion')):
            hashes = {}
            for n, k, b in handoff.CELLS:
                name = f'benchmark-vector-n{n}-k{k}-b{b}{suffix}'
                path = self.inputs / name
                path.write_text('Synthetic input for handoff tests: ' + name)
                hashes[name] = handoff.digest(path)
            (self.inputs / filename).write_text(json.dumps(hashes))
        self.rotation = self.base / 'rotation.bin'
        self.rotation.write_bytes(handoff.ROTATION_HEADER + b'Synthetic fixture, not native evidence')
        self.output = self.base / 'handoff'
        self.verification = self.base / 'verification'
        self.verification.mkdir()
        (self.verification / 'cpu.log').write_text('Synthetic offline record; not native evidence')

    def git(self, *args):
        return handoff.run(['git', *args], self.repo)

    def commit(self, message):
        self.git('-c', 'user.name=Handoff Test', '-c', 'user.email=handoff@example.invalid',
                 'commit', '-qm', message)

    def create(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return handoff.create_bundle(self.repo, self.inputs, self.rotation,
                                         self.output, 'rtx-4080-super', self.verification)

    def test_archive_restores_exact_source_and_plans_without_gpu(self):
        result = self.create()
        self.assertEqual(result['status'], 'OFFLINE_HANDOFF_READY')
        self.assertEqual(result['gpu_execution'], 'NOT_RUN')
        self.assertTrue(Path(result['archive']).is_file())
        manifest = handoff.verify_bundle(self.output)
        self.assertIn('verification/cpu.log', manifest['files'])
        self.assertEqual(manifest['source_commit'], self.git('rev-parse', 'HEAD'))
        self.assertEqual(result['files'], len(manifest['files']) + 2)
        portable = self.output / 'handoff.py'
        # The copied helper verifies the package without importing checkout modules.
        verified = json.loads(handoff.run([sys.executable, portable, 'verify', self.output], self.base))
        self.assertEqual(verified['gpu_execution'], 'NOT_RUN')
        checkout = self.base / 'restored'
        handoff.run([sys.executable, portable, 'restore', self.output, checkout], self.base)
        self.assertEqual(handoff.source_revision(checkout), manifest['source_commit'])
        plan = json.loads(handoff.run([sys.executable, portable, 'plan', self.output, checkout], self.base))
        self.assertEqual(plan['status'], 'PLANNED')
        self.assertFalse(plan['rentals_created'])
        self.assertTrue(plan['extended_probes'])
        native = plan['gpu_command_for_later']
        self.assertEqual(native[native.index('--mode') + 1], 'gpu')
        self.assertIn('--rotation-fixture', native)
        self.assertIn('--extended-probes', native)
        self.assertIn('--defer-analysis', native)
        self.assertEqual(plan['gpu_execution'], 'NOT_RUN')
        self.assertEqual(plan['target']['model'], 'RTX 4080 SUPER')
        self.assertEqual(plan['rotation_fixture_sha256'], handoff.digest(self.rotation))
        self.assertFalse((checkout / 'build').exists())
        with self.assertRaises(ValueError):
            self.create()

    def test_changed_missing_extra_or_linked_files_are_rejected(self):
        self.create()
        target = self.output / 'prepared' / next(iter(handoff.input_hashes(self.inputs)))
        original = target.read_bytes()
        target.write_bytes(original + b'corrupt')
        with self.assertRaisesRegex(ValueError, 'file changed'):
            handoff.verify_bundle(self.output)
        target.write_bytes(original)
        target.unlink()
        with self.assertRaisesRegex(ValueError, 'inventory changed'):
            handoff.verify_bundle(self.output)
        target.symlink_to(self.rotation)
        with self.assertRaisesRegex(ValueError, 'link or special'):
            handoff.verify_bundle(self.output)
        target.unlink(); target.write_bytes(original)
        extra = self.output / 'unexpected'
        extra.write_text('unexpected')
        with self.assertRaisesRegex(ValueError, 'inventory changed'):
            handoff.verify_bundle(self.output)
        extra.unlink()
        manifest_path = self.output / 'manifest.json'
        original_manifest = manifest_path.read_text()
        manifest = json.loads(original_manifest)
        manifest['target'] = 'rtx-5090'
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'checksums or manifest changed'):
            handoff.verify_bundle(self.output)
        manifest_path.write_text(original_manifest)
        self.assertEqual(handoff.verify_bundle(self.output)['target'], 'rtx-4080-super')

    def test_dirty_source_incomplete_matrix_and_bad_rotation_never_create_package(self):
        dirty = self.repo / 'uncommitted'
        dirty.write_text('pending')
        with self.assertRaisesRegex(ValueError, 'Commit source changes'):
            self.create()
        dirty.unlink()
        vectors = self.inputs / 'vectors.json'
        hashes = json.loads(vectors.read_text())
        saved = vectors.read_text()
        hashes.pop(next(iter(hashes)))
        vectors.write_text(json.dumps(hashes))
        with self.assertRaisesRegex(ValueError, 'eight rental workloads'):
            self.create()
        vectors.write_text(saved)
        self.rotation.write_bytes(b'not a rotation fixture')
        with self.assertRaisesRegex(ValueError, 'rotation fixture header'):
            self.create()
        self.assertFalse(self.output.exists())

    def test_restore_refuses_existing_directory_and_plan_refuses_other_revision(self):
        self.create()
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(handoff.main(['restore', str(self.output), str(self.repo)]), 1)
        (self.repo / 'new-version').write_text('next')
        self.git('add', '.'); self.commit('Different source')
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(handoff.main(['plan', str(self.output), str(self.repo)]), 1)


if __name__ == '__main__':
    unittest.main()
