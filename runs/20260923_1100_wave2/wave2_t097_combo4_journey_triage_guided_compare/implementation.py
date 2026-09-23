"""Synthetic, deterministic journey -> triage -> setup -> comparison CLI.

Input schema version 1; all fields are required and unknown fields are rejected.
Each stage receives and validates the same accumulating envelope. Completion is
user-supplied evidence, never inferred from a recommendation. No external calls.
"""
import copy
import json
import math
import sys


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def obj(value, fields, path):
    require(isinstance(value, dict), path + ' must be an object')
    require(set(value) == set(fields.split()), path + ' has missing or unknown fields')


def text(value, path):
    require(isinstance(value, str) and bool(value.strip()), path + ' must be nonempty text')


def array(value, path):
    require(isinstance(value, list), path + ' must be an array')


def strings(value, path):
    array(value, path)
    for item in value:
        text(item, path)
    require(len(set(value)) == len(value), path + ' has duplicates')


def number(value, path, positive=False):
    require(type(value) in (int, float) and math.isfinite(value), path + ' must be finite numeric')
    require(value > 0 if positive else value >= 0, path + ' is out of range')


def indexed(items, path):
    array(items, path)
    result = {}
    for item in items:
        require(isinstance(item, dict) and 'id' in item, path + ' item requires id')
        text(item['id'], path + '.id')
        require(item['id'] not in result, path + ' duplicate id')
        result[item['id']] = item
    return result


def graph(items, completed, dependency_key, path):
    for item in items.values():
        strings(item[dependency_key], path + '.requires')
        require(set(item[dependency_key]) <= items.keys(), path + ' unknown prerequisite')
    visiting, done = set(), set()

    def visit(key):
        require(key not in visiting, path + ' contains a prerequisite cycle')
        if key in done:
            return
        visiting.add(key)
        for dep in items[key][dependency_key]:
            visit(dep)
        visiting.remove(key)
        done.add(key)

    for key in items:
        visit(key)
    strings(completed, path + '.completed')
    require(set(completed) <= items.keys(), path + ' unknown completion')
    for key in completed:
        require(set(items[key][dependency_key]) <= set(completed), path + ' completion lacks prerequisites')


PRIORITIES = ('low', 'normal', 'high', 'urgent')
UNITS = {'price': {'USD': 1, 'cent': .01},
         'capacity': {'GB': 1, 'TB': 1000},
         'warranty': {'month': 1, 'year': 12}}
CANONICAL_UNITS = {'price': 'USD', 'capacity': 'GB', 'warranty': 'month'}
STAGES = ('journey', 'triage', 'guided', 'compare')


def validate_input(data):
    obj(data, 'schema_version synthetic profile actions ticket triage steps products preferences', 'input')
    require(type(data['schema_version']) is int and data['schema_version'] == 1, 'unsupported schema_version')
    require(data['synthetic'] is True, 'synthetic fixtures must be explicitly labeled')
    profile = data['profile']
    obj(profile, 'interests completed_actions completed_steps', 'profile')
    strings(profile['interests'], 'interests')
    actions = indexed(data['actions'], 'actions')
    for action in actions.values():
        obj(action, 'id tags requires', 'action')
        strings(action['tags'], 'action.tags')
    graph(actions, profile['completed_actions'], 'requires', 'actions')
    obj(data['ticket'], 'text impact', 'ticket')
    text(data['ticket']['text'], 'ticket.text')
    require(data['ticket']['impact'] in ('low', 'medium', 'high'), 'invalid ticket impact')
    config = data['triage']
    obj(config, 'rules default impact_priorities', 'triage')
    array(config['rules'], 'triage.rules')
    obj(config['impact_priorities'], 'low medium high', 'impact_priorities')
    for value in config['impact_priorities'].values():
        require(value in PRIORITIES, 'invalid impact priority')
    categories = set()
    for rule in config['rules'] + [config['default']]:
        obj(rule, 'category keywords priority owner' if rule is not config['default'] else 'category priority owner', 'routing rule')
        text(rule['category'], 'category')
        text(rule['owner'], 'owner')
        require(rule['priority'] in PRIORITIES, 'invalid routing priority')
        if 'keywords' in rule:
            strings(rule['keywords'], 'keywords')
            require(bool(rule['keywords']), 'rule keywords must not be empty')
        categories.add(rule['category'])
    steps = indexed(data['steps'], 'steps')
    for step in steps.values():
        obj(step, 'id action_id categories requires', 'step')
        require(isinstance(step['action_id'], str) and step['action_id'] in actions, 'step unknown action')
        strings(step['categories'], 'step.categories')
        require(bool(step['categories']) and set(step['categories']) <= categories | {'*'}, 'step unknown category')
    graph(steps, profile['completed_steps'], 'requires', 'steps')
    products = indexed(data['products'], 'products')
    for product in products.values():
        obj(product, 'id name attributes requires_steps', 'product')
        text(product['name'], 'product.name')
        strings(product['requires_steps'], 'product.requires_steps')
        require(set(product['requires_steps']) <= steps.keys(), 'product unknown setup prerequisite')
        obj(product['attributes'], 'price capacity warranty', 'attributes')
        for attr, measurement in product['attributes'].items():
            obj(measurement, 'value unit', attr)
            number(measurement['value'], attr)
            require(isinstance(measurement['unit'], str) and measurement['unit'] in UNITS[attr], 'unsupported ' + attr + ' unit')
            require(math.isfinite(measurement['value'] * UNITS[attr][measurement['unit']]), 'normalized attribute overflow')
    obj(data['preferences'], 'price capacity warranty', 'preferences')
    weights = []
    for attr, pref in data['preferences'].items():
        obj(pref, 'weight direction', 'preference.' + attr)
        number(pref['weight'], 'weight')
        require(pref['direction'] in ('min', 'max'), 'direction must be min or max')
        weights.append(pref['weight'])
    require(math.isfinite(sum(weights)) and sum(weights) > 0, 'total preference weight must be finite and positive')
    return data


def journey_result(data):
    actions = {a['id']: a for a in data['actions']}
    completed = set(data['profile']['completed_actions'])
    interests = set(data['profile']['interests'])
    score = lambda key: len(set(actions[key]['tags']) & interests)
    feasible = lambda done: [key for key, a in actions.items() if key not in done and set(a['requires']) <= done]
    next_ids = sorted(feasible(completed), key=lambda key: (-score(key), key))
    pairs = [(first, second) for first in next_ids for second in feasible(completed | {first})]
    require(bool(pairs), 'no valid two-step journey remains')
    pair = min(pairs, key=lambda pair: (-sum(score(key) for key in pair), pair))
    return {'next_actions': [{'action_id': key, 'interest_score': score(key)} for key in next_ids],
            'action_ids': list(pair), 'interest_score': sum(score(key) for key in pair)}


def triage_result(data, journey):
    ticket, config = data['ticket'], data['triage']
    message = ticket['text'].casefold()
    matches = [rule for rule in config['rules'] if any(word.casefold() in message for word in rule['keywords'])]
    rule = matches[0] if matches else config['default']
    impact_priority = config['impact_priorities'][ticket['impact']]
    priority = max((rule['priority'], impact_priority), key=PRIORITIES.index)
    return {'category': rule['category'], 'priority': priority, 'owner': rule['owner'],
            'journey_action_ids': list(journey['action_ids']),
            'matched_keywords': [word for word in rule.get('keywords', []) if word.casefold() in message],
            'rule_source': 'first_matching_rule' if matches else 'default'}


def guided_result(data, triage):
    steps = {step['id']: step for step in data['steps']}
    chosen = {key for key, step in steps.items()
              if step['action_id'] in triage['journey_action_ids']
              and (triage['category'] in step['categories'] or '*' in step['categories'])}
    pending = list(chosen)
    while pending:
        for dep in steps[pending.pop()]['requires']:
            if dep not in chosen:
                chosen.add(dep)
                pending.append(dep)
    completed = set(data['profile']['completed_steps'])
    order, remaining = [], set(chosen)
    while remaining:
        ready = sorted(key for key in remaining if not (set(steps[key]['requires']) & remaining))
        require(bool(ready), 'setup cycle')
        order.extend(ready)
        remaining.difference_update(ready)
    rows = []
    for key in order:
        missing = sorted(set(steps[key]['requires']) - completed)
        status = 'completed' if key in completed else ('blocked' if missing else 'available')
        rows.append({'step_id': key, 'status': status, 'missing_prerequisites': missing})
    done = len(chosen & completed)
    return {'owner': triage['owner'], 'category': triage['category'], 'priority': triage['priority'],
            'journey_action_ids': list(triage['journey_action_ids']), 'steps': rows,
            'completed_step_ids': sorted(completed), 'available_step_ids': [row['step_id'] for row in rows if row['status'] == 'available'],
            'progress': {'completed': done, 'total': len(chosen), 'fraction': done / len(chosen) if chosen else 1.0}}


def compare_result(data, guided):
    completed = set(guided['completed_step_ids'])
    rows = []
    for product in sorted(data['products'], key=lambda product: product['id']):
        missing = sorted(set(product['requires_steps']) - completed)
        attrs = {attr: item['value'] * UNITS[attr][item['unit']] for attr, item in product['attributes'].items()}
        rows.append({'product_id': product['id'], 'name': product['name'], 'attributes': attrs,
                     'eligible': not missing, 'missing_steps': missing})
    eligible = [row for row in rows if row['eligible']]
    total_weight = sum(pref['weight'] for pref in data['preferences'].values())
    ranked = []
    for row in eligible:
        utilities = {}
        for attr, pref in data['preferences'].items():
            values = [item['attributes'][attr] for item in eligible]
            low, high = min(values), max(values)
            utility = 1.0 if high == low else (row['attributes'][attr] - low) / (high - low)
            if high != low and pref['direction'] == 'min':
                utility = 1 - utility
            utilities[attr] = utility
        score = sum(utilities[attr] * (pref['weight'] / total_weight) for attr, pref in data['preferences'].items())
        ranked.append({'product_id': row['product_id'], 'score': score, 'utilities': utilities})
    ranked.sort(key=lambda item: (-item['score'], item['product_id']))
    return {'owner': guided['owner'], 'setup_progress': copy.deepcopy(guided['progress']),
            'units': dict(CANONICAL_UNITS), 'rows': rows, 'ranking': ranked,
            'selected_product_id': ranked[0]['product_id'] if ranked else None}


BUILDERS = (journey_result, triage_result, guided_result, compare_result)


def validate_envelope(envelope, count):
    obj(envelope, 'schema_version status input stages', 'envelope')
    require(type(envelope['schema_version']) is int and envelope['schema_version'] == 1, 'invalid envelope version')
    require(envelope['status'] == ('ok' if count == 4 else 'in_progress'), 'invalid envelope status')
    validate_input(envelope['input'])
    require(isinstance(envelope['stages'], dict) and set(envelope['stages']) == set(STAGES[:count]), 'invalid stage sequence')
    previous = None
    for index, stage in enumerate(STAGES[:count]):
        expected = BUILDERS[index](envelope['input'], previous) if index else BUILDERS[index](envelope['input'])
        require(envelope['stages'][stage] == expected, 'invalid validated output for ' + stage)
        previous = expected
    return envelope


def advance(envelope, stage):
    require(stage in STAGES, 'unknown stage')
    index = STAGES.index(stage)
    validate_envelope(envelope, index)
    result = copy.deepcopy(envelope)
    previous = result['stages'][STAGES[index - 1]] if index else None
    result['stages'][stage] = BUILDERS[index](result['input'], previous) if index else BUILDERS[index](result['input'])
    result['status'] = 'ok' if index == 3 else 'in_progress'
    return validate_envelope(result, index + 1)


def run(data):
    validate_input(data)
    envelope = {'schema_version': 1, 'status': 'in_progress', 'input': copy.deepcopy(data), 'stages': {}}
    for stage in STAGES:
        envelope = advance(envelope, stage)
    return envelope


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'duplicate JSON key: ' + key)
        result[key] = value
    return result


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    try:
        require(len(args) == 1, 'usage: python -B implementation.py INPUT.json')
        with open(args[0], encoding='utf-8') as source:
            data = json.load(source, object_pairs_hook=unique_object)
        result = run(data)
        encoded = json.dumps(result, allow_nan=False, sort_keys=True)
    except (OSError, UnicodeError, ValueError, TypeError, OverflowError, RecursionError) as error:
        print(json.dumps({'status': 'error', 'error': str(error)}))
        return 2
    print(encoded)
    return 0


if __name__ == '__main__':
    sys.exit(main())
