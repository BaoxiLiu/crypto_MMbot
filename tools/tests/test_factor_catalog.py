"""Inventory integrity, unsafe selection prevention, and source identity checks."""
import hashlib
from pathlib import Path
import tempfile
import unittest

from app.factor_library.catalog import load_catalog, lookup, select, verify_sources


class FactorCatalogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_catalog()

    def test_inventory_matches_all_seven_source_collections(self):
        data = self.data
        self.assertEqual(len(data['factors']), 602)
        self.assertEqual(sum(c['count'] for c in data['collections'].values()), 602)
        for key, collection in data['collections'].items():
            rows = [r for r in data['factors'] if r['collection'] == key]
            self.assertEqual(len(rows), collection['count'], key)
        self.assertEqual(len(data['categories']), 9)
        self.assertEqual(sum(data['historical_category_counts'].values()), 523)

    def test_default_selection_excludes_candidates_and_blocked_backends(self):
        rows = select(catalog=self.data)
        self.assertEqual(len(rows), 523)
        self.assertTrue(all(r['status']=='implemented_source' for r in rows))
        self.assertEqual(select(collection='cicc', catalog=self.data), [])
        self.assertEqual(len(select(collection='paper', catalog=self.data)),173)
        self.assertEqual(len(select(collection='paper', include_blocked=True, catalog=self.data)),173)
        blocked={**self.data,'factors':[{**r,'status':'blocked_missing_dependency'}
                                      if r['collection']=='paper' else r for r in self.data['factors']]}
        self.assertEqual(select(collection='paper',catalog=blocked),[])

    def test_candidates_must_be_explicit_and_unresolved_formulas_stay_empty(self):
        rows = select(collection='cicc', include_candidates=True, catalog=self.data)
        self.assertEqual(len(rows),79)
        unresolved = [r for r in rows if r['status']=='formula_unresolved']
        self.assertEqual(len(unresolved),36)
        self.assertTrue(all(r['formula'] is None for r in unresolved))
        self.assertTrue(all(r['implementation'] is None for r in rows))

    def test_namespaces_prevent_similar_names_becoming_automatic_aliases(self):
        with self.assertRaises(KeyError): lookup('spread_bp',catalog=self.data)
        self.assertEqual(lookup('micro:spread_bp',catalog=self.data)['collection'],'micro')
        self.assertNotEqual(lookup('v3:C4',catalog=self.data)['clock'],
                            lookup('paper:trade_intensity_1s_w0',catalog=self.data)['clock'])
        self.assertIn('NOT trades/second',lookup('paper:trade_intensity_1s_w0',catalog=self.data)['formula'])

    def test_reused_diagnostic_columns_resolve_without_duplicate_entries(self):
        ids={r['id'] for r in self.data['factors']}
        self.assertEqual(len(self.data['known_reuse']),80)
        for alias in self.data['known_reuse']:
            self.assertNotIn(alias['alias'],ids)
            self.assertEqual(lookup(alias['alias'],catalog=self.data)['id'],alias['target'])

    def test_historical_segments_and_windows_remain_explicit(self):
        a=lookup('paper:ret_sum_1s_w0',catalog=self.data)
        b=lookup('paper:ret_sum_1s_w2',catalog=self.data)
        self.assertEqual(a['family_key'],b['family_key'])
        self.assertNotEqual(a['parameters'],b['parameters'])
        self.assertEqual(b['parameters'],{'scale':'1s','historical_segment':2})

    def test_invalid_filters_fail_instead_of_silently_returning_wrong_scope(self):
        with self.assertRaises(ValueError): select(category='future_markout',catalog=self.data)
        with self.assertRaises(ValueError): select(collection='all523',catalog=self.data)

    def test_source_drift_and_new_dependency_require_explicit_rebuild(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); p=root/'definition.py';p.write_text('old')
            d={'source_sha256':{'definition.py':hashlib.sha256(p.read_bytes()).hexdigest()},
               'missing_sources':['dependency.csv']}
            self.assertTrue(verify_sources(catalog=d,root=root))
            p.write_text('new')
            with self.assertRaisesRegex(ValueError,'changed or missing'):verify_sources(catalog=d,root=root)
            p.write_text('old');(root/'dependency.csv').write_text('now present')
            with self.assertRaisesRegex(ValueError,'dependency appeared'):verify_sources(catalog=d,root=root)

    def test_current_source_fingerprints(self):
        self.assertTrue(verify_sources(catalog=self.data))


if __name__=='__main__':unittest.main()
