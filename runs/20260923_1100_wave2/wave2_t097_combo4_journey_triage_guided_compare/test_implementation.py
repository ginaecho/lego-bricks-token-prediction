"""Tests use only explicitly synthetic fixtures; no external dependencies."""
import contextlib
import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import mock_open, patch

import implementation as app

ROOT = Path(__file__).resolve().parent


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / 'example_input.json').read_text(encoding='utf-8'))

    def run_pipeline(self):
        return app.run(self.data)['stages']

    def invalid(self):
        with self.assertRaises(app.ValidationError):
            app.run(self.data)

    def test_integrated_order_and_success(self):
        output = app.run(self.data)
        self.assertEqual(output['status'], 'ok')
        self.assertEqual(list(output['stages']), list(app.STAGES))
        self.assertIs(app.validate_envelope(output, 4), output)

    def test_journey_prerequisites_and_personalization(self):
        journey = self.run_pipeline()['journey']
        self.assertEqual(journey['action_ids'], ['configure', 'evaluate'])
        self.assertEqual([x['action_id'] for x in journey['next_actions']], ['configure', 'learn'])
        self.assertEqual(journey['interest_score'], 2)

    def test_two_step_lookahead(self):
        self.data['actions'].append({'id': 'premium', 'tags': ['storage', 'value'], 'requires': ['learn']})
        self.assertEqual(self.run_pipeline()['journey']['action_ids'], ['configure', 'evaluate'])
        self.data['actions'][1]['tags'] = []
        self.assertEqual(self.run_pipeline()['journey']['action_ids'], ['learn', 'premium'])

    def test_journey_no_two_steps(self):
        self.data['profile']['completed_actions'] = ['discover', 'configure', 'evaluate']
        self.invalid()

    def test_journey_cycle(self):
        self.data['actions'][0]['requires'] = ['evaluate']
        self.invalid()

    def test_unknown_action_prerequisite(self):
        self.data['actions'][1]['requires'] = ['absent']
        self.invalid()

    def test_inconsistent_action_completion(self):
        self.data['profile']['completed_actions'] = ['evaluate']
        self.invalid()

    def test_triage_case_insensitive_match_and_impact(self):
        self.data['ticket']['text'] = 'Synthetic CONFIGURE problem'
        triage = self.run_pipeline()['triage']
        self.assertEqual(triage['category'], 'setup')
        self.assertEqual(triage['priority'], 'urgent')
        self.assertEqual(triage['owner'], 'synthetic-onboarding-team')

    def test_triage_first_matching_rule(self):
        self.data['ticket']['text'] = 'Synthetic invoice configure'
        self.assertEqual(self.run_pipeline()['triage']['category'], 'setup')

    def test_triage_default(self):
        self.data['ticket'] = {'text': 'Synthetic unknown topic', 'impact': 'low'}
        triage = self.run_pipeline()['triage']
        self.assertEqual(triage['category'], 'general')
        self.assertEqual(triage['priority'], 'low')
        self.assertEqual(triage['rule_source'], 'default')

    def test_rule_priority_floor(self):
        self.data['ticket'] = {'text': 'Synthetic invoice', 'impact': 'low'}
        self.assertEqual(self.run_pipeline()['triage']['priority'], 'high')

    def test_configurable_impact(self):
        self.data['triage']['impact_priorities']['high'] = 'high'
        self.assertEqual(self.run_pipeline()['triage']['priority'], 'high')

    def test_unaccountable_route_rejected(self):
        self.data['triage']['rules'][0]['owner'] = ' '
        self.invalid()

    def test_all_handoffs_propagate(self):
        stages = self.run_pipeline()
        self.assertEqual(stages['journey']['action_ids'], stages['triage']['journey_action_ids'])
        self.assertEqual(stages['triage']['journey_action_ids'], stages['guided']['journey_action_ids'])
        self.assertEqual(stages['triage']['owner'], stages['guided']['owner'])
        self.assertEqual(stages['triage']['priority'], stages['guided']['priority'])
        self.assertEqual(stages['guided']['owner'], stages['compare']['owner'])
        self.assertEqual(stages['guided']['progress'], stages['compare']['setup_progress'])

    def test_guided_closure_and_status(self):
        guided = self.run_pipeline()['guided']
        self.assertEqual(guided['steps'], [
            {'step_id': 'account', 'status': 'completed', 'missing_prerequisites': []},
            {'step_id': 'connect', 'status': 'available', 'missing_prerequisites': []},
            {'step_id': 'verify', 'status': 'blocked', 'missing_prerequisites': ['connect']}])
        self.assertEqual(guided['progress'], {'completed': 1, 'total': 3, 'fraction': 1/3})

    def test_progress_update_changes_comparison(self):
        self.data['profile']['completed_steps'] = ['account', 'connect', 'verify']
        stages = self.run_pipeline()
        self.assertEqual(stages['guided']['progress']['fraction'], 1)
        self.assertEqual(len(stages['compare']['ranking']), 3)
        self.assertTrue(all(row['status'] == 'completed' for row in stages['guided']['steps']))

    def test_triage_changes_guided_selection(self):
        self.data['ticket']['text'] = 'Synthetic invoice'
        guided = self.run_pipeline()['guided']
        self.assertEqual(guided['steps'], [])
        self.assertEqual(guided['progress']['fraction'], 1)
        self.assertEqual(guided['owner'], 'synthetic-billing-team')

    def test_inconsistent_step_completion(self):
        self.data['profile']['completed_steps'] = ['verify']
        self.invalid()

    def test_step_cycle(self):
        self.data['steps'][0]['requires'] = ['verify']
        self.invalid()

    def test_normalization_and_comparison(self):
        comparison = self.run_pipeline()['compare']
        basic = comparison['rows'][0]
        self.assertEqual(basic['attributes'], {'price': 100, 'capacity': 500, 'warranty': 12})
        self.assertEqual(comparison['units'], {'price': 'USD', 'capacity': 'GB', 'warranty': 'month'})
        self.assertEqual(comparison['selected_product_id'], 'basic')
        self.assertAlmostEqual(comparison['ranking'][0]['score'], .6)
        self.assertFalse(comparison['rows'][2]['eligible'])

    def test_preference_changes_ranking(self):
        self.data['preferences']['price']['weight'] = 0
        self.assertEqual(self.run_pipeline()['compare']['selected_product_id'], 'plus')

    def test_equal_attributes_tie_break_by_id(self):
        self.data['products'][1]['attributes'] = copy.deepcopy(self.data['products'][0]['attributes'])
        ranked = self.run_pipeline()['compare']['ranking']
        self.assertEqual([row['product_id'] for row in ranked], ['basic', 'plus'])
        self.assertAlmostEqual(ranked[0]['score'], 1)

    def test_no_eligible_products(self):
        self.data['profile']['completed_steps'] = []
        result = self.run_pipeline()['compare']
        self.assertEqual(result['ranking'], [])
        self.assertIsNone(result['selected_product_id'])

    def test_empty_catalog(self):
        self.data['products'] = []
        self.assertEqual(self.run_pipeline()['compare']['rows'], [])

    def test_unknown_unit(self):
        self.data['products'][0]['attributes']['price']['unit'] = 'EUR'
        self.invalid()

    def test_invalid_numeric_values(self):
        for value in (True, '100', -1, float('nan'), float('inf')):
            with self.subTest(value=value):
                self.data['products'][0]['attributes']['price']['value'] = value
                self.invalid()

    def test_missing_attribute(self):
        del self.data['products'][0]['attributes']['warranty']
        self.invalid()

    def test_zero_weights(self):
        for pref in self.data['preferences'].values():
            pref['weight'] = 0
        self.invalid()

    def test_unknown_product_prerequisite(self):
        self.data['products'][0]['requires_steps'] = ['absent']
        self.invalid()

    def test_duplicate_ids(self):
        self.data['products'].append(copy.deepcopy(self.data['products'][0]))
        self.invalid()

    def test_unknown_and_missing_fields(self):
        self.data['extra'] = 'unknown'
        self.invalid()
        del self.data['extra']
        del self.data['profile']
        self.invalid()

    def test_bad_version_and_synthetic_label(self):
        self.data['schema_version'] = True
        self.invalid()
        self.data['schema_version'] = 1
        self.data['synthetic'] = False
        self.invalid()

    def test_no_mutation_and_deterministic(self):
        before = copy.deepcopy(self.data)
        first, second = app.run(self.data), app.run(self.data)
        self.assertEqual(self.data, before)
        self.assertEqual(first, second)

    def test_tampered_handoff_is_rejected(self):
        envelope = {'schema_version': 1, 'status': 'in_progress', 'input': self.data, 'stages': {}}
        envelope = app.advance(envelope, 'journey')
        envelope['stages']['journey']['action_ids'] = ['evaluate', 'configure']
        with self.assertRaises(app.ValidationError):
            app.advance(envelope, 'triage')

    def test_out_of_order_stage_rejected(self):
        envelope = {'schema_version': 1, 'status': 'in_progress', 'input': self.data, 'stages': {}}
        with self.assertRaises(app.ValidationError):
            app.advance(envelope, 'guided')

    def cli(self, *args):
        result = subprocess.run([sys.executable, '-B', str(ROOT / 'implementation.py'), *args], cwd=ROOT,
                                capture_output=True, text=True, timeout=15)
        self.assertEqual(result.stderr, '')
        payload = json.loads(result.stdout)
        self.assertEqual(len(result.stdout.strip().splitlines()), 1)
        return result.returncode, payload

    def test_cli_success(self):
        code, output = self.cli('example_input.json')
        self.assertEqual(code, 0)
        self.assertEqual(output['status'], 'ok')

    def test_cli_missing_file(self):
        code, output = self.cli('nonexistent-fixture.json')
        self.assertEqual(code, 2)
        self.assertEqual(output['status'], 'error')

    def test_cli_invalid_json_file(self):
        code, output = self.cli('test_implementation.py')
        self.assertEqual(code, 2)
        self.assertEqual(output['status'], 'error')

    def test_cli_usage(self):
        self.assertEqual(self.cli()[0], 2)
        self.assertEqual(self.cli('one', 'two')[0], 2)

    def test_cli_malformed_and_duplicate_json(self):
        for raw in ('[]', '{"schema_version":1,"schema_version":1}', '{broken', '{"x":NaN}'):
            output = io.StringIO()
            with self.subTest(raw=raw), patch('builtins.open', mock_open(read_data=raw)), contextlib.redirect_stdout(output):
                code = app.main(['synthetic.json'])
            self.assertEqual(code, 2)
            self.assertEqual(json.loads(output.getvalue())['status'], 'error')

    def test_cli_unicode_error(self):
        output = io.StringIO()
        with patch('builtins.open', side_effect=UnicodeError('synthetic decoding failure')), contextlib.redirect_stdout(output):
            self.assertEqual(app.main(['synthetic.json']), 2)
        self.assertEqual(json.loads(output.getvalue())['status'], 'error')


if __name__ == '__main__':
    unittest.main()
