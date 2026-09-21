from __future__ import annotations
import gzip, hashlib, json, math, statistics, sys, zipfile
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path

ARTIFACT = Path(r'C:\dev\trade_rl\.tmp\g4-audit-10614569766-20260922\artifact.zip')
REPORT = ARTIFACT.parent / 'g4-audit.json'
EXPECTED_SIZE = 589_319_909
EXPECTED_SHA = '3997d64142c9143085ea954ef340905c4764ce58d264e21ade791b8d41f7120f'
RUN_ID = 35_542_548_393
ARTIFACT_ID = 10_614_569_766
HEAD = 'b876f1c5b5a7b3b350b625f853dc72a48f3c1530'
CHECKPOINT_PROTOCOL = 'e04d0146fea39bdbb95e1b78ed5b94b2fead296f67774f1007f0496236e9b5d2'
CORE_PROTOCOL = 'e37701b92ddfb93f3bc6d528e1affebcd692db929415fa1b27ff65ea94dbd475'
G3_AUDIT_SHA = '3a2b8d353aa4e176e0dacb1bf9963cfb2995e905d51611b37641f96d5fbf8e5a'
FACTORS = ('baseline', 'btc_relative')
SCENARIOS = {'baseline': ('base',), 'btc_relative': ('base', 'cost_2x', 'latency_1')}
SYMBOLS = ('BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT')
SEEDS = tuple(range(5))
TOL = 1e-10

def need(ok: bool, message: str) -> None:
    if not ok:
        raise AssertionError(message)

def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()

def digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def jread(z: zipfile.ZipFile, name: str):
    return json.loads(z.read(name))

def close(a: float, b: float, *, abs_tol: float = 1e-9) -> bool:
    return math.isclose(float(a), float(b), rel_tol=1e-10, abs_tol=abs_tol)

def compound(values) -> float:
    vals = tuple(float(x) for x in values)
    need(bool(vals) and all(math.isfinite(x) and x > -1.0 for x in vals), 'invalid return sequence')
    return math.expm1(math.fsum(math.log1p(x) for x in vals))

def finite(x, field: str) -> float:
    need(not isinstance(x, bool) and isinstance(x, (int, float)) and math.isfinite(float(x)), f'{field} is not finite numeric')
    return float(x)

def deep_equal(a, b, path='comparison', differences=None):
    if differences is None:
        differences = []
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            differences.append(f'{path}: key set differs')
            return differences
        for key in a:
            deep_equal(a[key], b[key], f'{path}.{key}', differences)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            differences.append(f'{path}: list length differs')
            return differences
        for i, (x, y) in enumerate(zip(a, b)):
            deep_equal(x, y, f'{path}[{i}]', differences)
    elif isinstance(a, (int, float)) and not isinstance(a, bool) and isinstance(b, (int, float)) and not isinstance(b, bool):
        if not close(float(a), float(b), abs_tol=2e-10):
            differences.append(f'{path}: {a!r} != {b!r}')
    elif a != b:
        differences.append(f'{path}: {a!r} != {b!r}')
    return differences

def main() -> int:
    need(ARTIFACT.stat().st_size == EXPECTED_SIZE, 'download byte count does not match G3 identity')
    artifact_sha = digest_file(ARTIFACT)
    need(artifact_sha == EXPECTED_SHA, 'download SHA-256 does not match G3 identity')
    with zipfile.ZipFile(ARTIFACT) as z:
        names = z.namelist()
        need(len(names) == 487 and len(names) == len(set(names)), 'artifact ZIP topology differs from G3 roster')
        receipt = jread(z, 'receipt.json')
        need(receipt['run_id'] == RUN_ID and receipt['run_attempt'] == 1, 'receipt run identity mismatch')
        need(receipt['artifact_name'] == f'ppo-feature-checkpoint-{RUN_ID}-1', 'receipt artifact name mismatch')
        need(receipt['code_sha'] == HEAD and receipt['github_sha'] == HEAD, 'receipt source head mismatch')
        need(receipt['stage'] == 'finalize' and receipt['status'] == 'succeeded', 'receipt is not successful finalize')
        need(receipt['checkpoint_protocol_digest'] == CHECKPOINT_PROTOCOL, 'receipt protocol digest mismatch')
        protocol = jread(z, 'checkpoint/checkpoint-protocol.json')
        protocol_sidecar = jread(z, 'checkpoint/checkpoint-protocol.digest.json')
        core = protocol['core_protocol']
        need(protocol_sidecar == {'digest': CHECKPOINT_PROTOCOL}, 'checkpoint protocol sidecar mismatch')
        need(protocol['core_protocol_digest'] == CORE_PROTOCOL, 'core protocol digest mismatch')
        need(sha(json.dumps(core, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(',', ':')).encode('utf-8')) == CORE_PROTOCOL, 'canonical core protocol hash mismatch')
        need(sha(json.dumps(protocol, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(',', ':')).encode('utf-8')) == CHECKPOINT_PROTOCOL, 'canonical checkpoint protocol hash mismatch')
        need(core['seeds'] == list(SEEDS), 'seed roster mismatch')
        need(core['symbols'] == list(SYMBOLS), 'symbol roster mismatch')
        need(core['training']['requested_timesteps'] == 262144, 'PPO step budget mismatch')
        ev = core['evaluation']
        need(ev['interval_count'] == 17544, 'evaluation interval count mismatch')
        need(ev['interval_year_slices'] == {'2023': [0, 8760], '2024': [8760, 17544]}, 'registered year slices mismatch')
        need(ev['stress_names'] == ['cost_2x', 'latency_1'], 'stress roster mismatch')
        need(ev['ledger_drawdown_limit'] == 0.2, 'drawdown limit mismatch')
        admission = core['admission']
        need(admission['required_passing_seeds_per_symbol'] == 4 and admission['required_symbols'] == 4 and admission['required_positive_paired_seed_deltas'] == 4, 'admission thresholds mismatch')
        comparison_raw = z.read('checkpoint/comparison/comparison.json')
        source_comparison = json.loads(comparison_raw)
        source_comparison_sidecar = jread(z, 'checkpoint/comparison/comparison.digest.json')
        need(source_comparison_sidecar == {'digest': sha(comparison_raw)}, 'source comparison sidecar mismatch')

        rows = {factor: {seed: {'base': {}, 'stress': {}} for seed in SEEDS} for factor in FACTORS}
        cell_summaries = []
        max_return_row_error = 0.0
        max_dd_endpoint = 0.0
        max_year_diagnostic_gap = 0.0
        total_fills = 0
        expected_cells = 0
        for factor in FACTORS:
            for seed in SEEDS:
                arm_path = f'checkpoint/arms/{factor}/ppo{seed}/result.json'
                arm = jread(z, arm_path)
                need(arm['factor'] == factor and arm['seed'] == seed, f'arm identity mismatch {factor}/{seed}')
                expected_timesteps = arm['actual_timesteps']
                need(expected_timesteps == 262144, f'actual PPO steps mismatch {factor}/{seed}')
                for scenario in SCENARIOS[factor]:
                    for symbol_index, symbol in enumerate(SYMBOLS):
                        expected_cells += 1
                        root = f'checkpoint/cells/{factor}/ppo{seed}/{scenario}/symbol-{symbol_index}'
                        cell = jread(z, root + '/cell.json')
                        need((cell['factor'], cell['seed'], cell['scenario'], cell['symbol_index'], cell['symbol']) == (factor, seed, scenario, symbol_index, symbol), f'cell identity mismatch {factor}/{seed}/{scenario}/{symbol}')
                        need(cell['checkpoint_protocol_digest'] == CHECKPOINT_PROTOCOL and cell['core_protocol_digest'] == CORE_PROTOCOL, f'cell protocol binding mismatch {factor}/{seed}/{symbol}')
                        replay = cell['replay']
                        ledger_name = replay['ledger_evidence_file']
                        packed = z.read(root + '/' + ledger_name)
                        need(len(packed) == cell['ledger_size_bytes'], f'ledger size mismatch {factor}/{seed}/{scenario}/{symbol}')
                        need(sha(packed) == replay['ledger_evidence_gzip_sha256'], f'ledger digest mismatch {factor}/{seed}/{scenario}/{symbol}')
                        copy_name = f'checkpoint/arms/{factor}/ppo{seed}/{ledger_name}'
                        need(z.read(copy_name) == packed, f'assembled ledger copy mismatch {factor}/{seed}/{scenario}/{symbol}')
                        raw = gzip.decompress(packed)
                        need(sha(raw) == replay['ledger_evidence_raw_sha256'], f'uncompressed ledger digest mismatch {factor}/{seed}/{scenario}/{symbol}')
                        ledger = json.loads(raw)
                        need(ledger['schema_version'] == 'shared_cash_replay_ledger_v1', 'ledger schema mismatch')
                        need(ledger['dataset_id'] == core['evaluation_dataset_id'], 'ledger dataset mismatch')
                        need(ledger['execution_policy_digest'] == ev['scenarios'][scenario]['execution_policy_digest'], 'ledger execution policy mismatch')
                        intervals = ledger['intervals']
                        need(len(intervals) == ev['interval_count'], 'ledger clock incomplete')
                        derived_returns = []
                        endpoint_values = [float(ev['initial_capital'])]
                        previous = None
                        fill_count = 0
                        filled_notional = 0.0
                        for offset, item in enumerate(intervals):
                            idx = ev['start_index'] + offset
                            need(item['start_index'] == idx and item['next_index'] == idx + 1, 'ledger interval index discontinuity')
                            before = finite(item['portfolio_value_before'], 'PV before')
                            after = finite(item['portfolio_value_after'], 'PV after')
                            need(before > 0 and after > 0, 'nonpositive account equity')
                            if previous is None:
                                need(before == ev['initial_capital'] and float(item['cash_before']) == ev['initial_capital'], 'initial account equity/cash mismatch')
                                need(not any(Fraction(x) for x in item['exact_quantities_before']), 'initial position is nonzero')
                                for f in ('total_cost_before', 'funding_pnl_before', 'borrow_cost_before', 'turnover_total_before', 'max_drawdown_before'):
                                    need(close(item[f], 0.0), f'nonzero initial cumulative {f}')
                            else:
                                for b, a in (('portfolio_value_before','portfolio_value_after'),('cash_before','cash_after'),('total_cost_before','total_cost_after'),('funding_pnl_before','funding_pnl_after'),('borrow_cost_before','borrow_cost_after'),('turnover_total_before','turnover_total_after'),('max_drawdown_before','max_drawdown_after')):
                                    need(close(item[b], previous[a]), f'ledger chain break {b}/{a}')
                                need([Fraction(x) for x in item['exact_quantities_before']] == [Fraction(x) for x in previous['exact_quantities_after']], 'exact quantity chain break')
                            r = after / before - 1.0
                            need(r > -1.0 and math.isfinite(r), 'invalid equity-derived interval return')
                            need(close(r, item['interval_net_return'], abs_tol=2e-10), 'ledger return disagrees with equity transition')
                            need(offset < len(replay['returns']) and close(r, replay['returns'][offset], abs_tol=2e-10), 'cell return disagrees with equity transition')
                            derived_returns.append(r)
                            endpoint_values.append(after)
                            for cf, bf, af in (('interval_cost','total_cost_before','total_cost_after'),('interval_funding','funding_pnl_before','funding_pnl_after'),('interval_borrow_cost','borrow_cost_before','borrow_cost_after')):
                                need(close(float(item[af])-float(item[bf]), float(item[cf])), f'{cf} cumulative delta mismatch')
                            for event in item.get('order_events', []):
                                need(event['execution_policy_digest'] == ev['scenarios'][scenario]['execution_policy_digest'], 'order event policy mismatch')
                                if abs(float(event['filled_quantity'])) > 0.0:
                                    fill_count += 1
                                    filled_notional += float(event['filled_notional'])
                            total_fills += sum(1 for event in item.get('order_events', []) if abs(float(event['filled_quantity'])) > 0.0)
                            previous = item
                        need(len(replay['returns']) == len(derived_returns), 'cell return series length mismatch')
                        full_return = compound(derived_returns)
                        need(close(full_return, float(replay['metrics']['total_return']), abs_tol=2e-10), 'equity-derived full return differs from cell metrics')
                        yearly = {year: compound(derived_returns[a:b]) for year, (a,b) in ev['interval_year_slices'].items()}
                        for year, value in yearly.items():
                            if year in replay.get('year_returns', {}):
                                max_year_diagnostic_gap = max(max_year_diagnostic_gap, abs(value-float(replay['year_returns'][year])))
                        peak = endpoint_values[0]
                        endpoint_dd = 0.0
                        for equity in endpoint_values[1:]:
                            peak = max(peak, equity)
                            endpoint_dd = max(endpoint_dd, 1.0 - equity / peak)
                        ledger_dd = finite(ledger['final_max_drawdown'], 'final ledger drawdown')
                        need(0.0 <= ledger_dd <= 1.0 and ledger_dd + 1e-11 >= endpoint_dd, 'ledger drawdown understates endpoint drawdown')
                        for item in intervals:
                            before_dd = finite(item['max_drawdown_before'], 'drawdown before')
                            after_dd = finite(item['max_drawdown_after'], 'drawdown after')
                            need(0.0 <= before_dd <= after_dd <= 1.0, 'invalid monotone drawdown trace')
                        need(close(ledger_dd, previous['max_drawdown_after']), 'final drawdown differs from interval trace')
                        need(close(ledger['final_portfolio_value'], previous['portfolio_value_after']), 'final equity differs from last interval')
                        need(close(ledger['final_cash'], previous['cash_after']), 'final cash differs from last interval')
                        need(close(ledger['final_total_cost'], previous['total_cost_after']), 'final cost differs from last interval')
                        need(close(ledger['final_turnover_total'], previous['turnover_total_after']), 'final turnover differs from last interval')
                        need(len(ledger['terminal_exact_quantities']) == len(SYMBOLS), 'terminal quantity vector length mismatch')
                        terminal_q = [Fraction(x) for x in ledger['terminal_exact_quantities']]
                        flat = all(abs(float(x)) <= 1e-10 for x in terminal_q)
                        need(flat == replay['terminal_flat'], 'exact terminal quantities disagree with flat flag')
                        need(ledger['terminal_exact_quantities'] == previous['exact_quantities_after'], 'terminal quantities differ from trace')
                        need(bool(ledger['active_order_remainders']) == bool(replay.get('active_order_remainders', [])), 'active remainder evidence differs')
                        termination = ledger['termination_reason']
                        need(termination == previous['termination_reason'], 'termination evidence differs from last interval')
                        term_reasons = replay['termination_reasons']
                        need((not term_reasons) == (termination is None), 'termination list differs from ledger')
                        need(close(replay['ledger_max_drawdown'], ledger_dd), 'cell drawdown differs from ledger')
                        need(close(replay['metrics']['max_drawdown'], endpoint_dd, abs_tol=2e-10), 'metric drawdown differs from equity path')
                        need(replay['metrics']['termination_count'] == (0 if termination is None else 1), 'termination count mismatch')
                        need(replay == (arm['by_symbol'][symbol] if scenario == 'base' else arm['stress_by_name'][scenario][symbol]), 'arm result row differs from cell manifest')
                        max_return_row_error = max(max_return_row_error, abs(full_return - float(replay['metrics']['total_return'])))
                        max_dd_endpoint = max(max_dd_endpoint, endpoint_dd)
                        summary = {
                            'factor': factor, 'seed': seed, 'scenario': scenario, 'symbol': symbol,
                            'total_return': full_return, 'year_returns': yearly,
                            'ledger_max_drawdown': ledger_dd, 'endpoint_drawdown': endpoint_dd,
                            'final_equity': finite(ledger['final_portfolio_value'], 'final equity'),
                            'total_cost': finite(ledger['final_total_cost'], 'total cost'),
                            'turnover': finite(ledger['final_turnover_total'], 'turnover'),
                            'funding_pnl': finite(ledger['final_funding_pnl'], 'funding PnL'),
                            'borrow_cost': finite(ledger['final_borrow_cost'], 'borrow cost'),
                            'terminal_flat': flat, 'termination_reason': termination,
                            'active_order_remainders': ledger['active_order_remainders'],
                            'fill_event_count': fill_count, 'filled_notional': filled_notional,
                        }
                        cell_summaries.append(summary)
                        bucket = rows[factor][seed]
                        bucket['base' if scenario == 'base' else 'stress'][symbol if scenario == 'base' else f'{scenario}:{symbol}'] = summary

        need(expected_cells == 100, 'expected replay cell count mismatch')
        # Recompute preregistered gates from the ledger-derived cell metrics.
        def get(factor, seed, scenario, symbol):
            key = symbol if scenario == 'base' else f'{scenario}:{symbol}'
            return rows[factor][seed]['base' if scenario == 'base' else 'stress'][key]
        def seed_pass(row):
            return (row['termination_reason'] is None and row['terminal_flat'] and not row['active_order_remainders'] and row['ledger_max_drawdown'] <= 0.20 and row['total_return'] > 0.0 and all(v > 0.0 for v in row['year_returns'].values()))
        hard_violations = []
        for seed in SEEDS:
            for symbol in SYMBOLS:
                for scenario in ('base', 'cost_2x', 'latency_1'):
                    row = get('btc_relative', seed, scenario, symbol)
                    reasons = []
                    if row['ledger_max_drawdown'] > 0.20: reasons.append('ledger_drawdown_exceeded')
                    if row['termination_reason'] is not None: reasons.append('terminated')
                    if not row['terminal_flat']: reasons.append('terminal_position_not_flat')
                    if row['active_order_remainders']: reasons.append('active_order_remainder')
                    if reasons: hard_violations.append({'seed':seed,'symbol':symbol,'scenario':scenario,'reasons':reasons})
        symbol_results = {}
        absolute_symbols, relative_symbols, stress_symbols = [], [], []
        for symbol in SYMBOLS:
            old = {seed: get('baseline', seed, 'base', symbol) for seed in SEEDS}
            new = {seed: get('btc_relative', seed, 'base', symbol) for seed in SEEDS}
            absolute_seed_pass = {seed: seed_pass(new[seed]) for seed in SEEDS}
            med_full = statistics.median(new[s]['total_return'] for s in SEEDS)
            med_year = {year: statistics.median(new[s]['year_returns'][year] for s in SEEDS) for year in ev['interval_year_slices']}
            absolute = (sum(absolute_seed_pass.values()) >= 4 and med_full > 0.0 and all(v > 0.0 for v in med_year.values()))
            if absolute: absolute_symbols.append(symbol)
            deltas = {s: new[s]['total_return']-old[s]['total_return'] for s in SEEDS}
            year_deltas = {year:{s:new[s]['year_returns'][year]-old[s]['year_returns'][year] for s in SEEDS} for year in ev['interval_year_slices']}
            comparable = {s:(old[s]['termination_reason'] is None and new[s]['termination_reason'] is None and old[s]['terminal_flat'] and new[s]['terminal_flat'] and not old[s]['active_order_remainders'] and not new[s]['active_order_remainders'] and new[s]['ledger_max_drawdown'] <= 0.20) for s in SEEDS}
            positive_seeds = [s for s in SEEDS if comparable[s] and deltas[s] > 0.0]
            med_delta = statistics.median(deltas.values())
            med_year_delta = {year:statistics.median(vals.values()) for year,vals in year_deltas.items()}
            relative = (len(positive_seeds) >= 4 and med_delta > 0.0 and all(v > 0.0 for v in med_year_delta.values()))
            if relative: relative_symbols.append(symbol)
            stress_details = {}
            stress_seed_pass = {seed:True for seed in SEEDS}
            for scenario in ('cost_2x','latency_1'):
                stress_rows={seed:get('btc_relative',seed,scenario,symbol) for seed in SEEDS}
                for seed in SEEDS: stress_seed_pass[seed] = stress_seed_pass[seed] and seed_pass(stress_rows[seed])
                stress_median_full=statistics.median(stress_rows[s]['total_return'] for s in SEEDS)
                stress_median_year={year:statistics.median(stress_rows[s]['year_returns'][year] for s in SEEDS) for year in ev['interval_year_slices']}
                stress_details[scenario]={'passing_seeds':[s for s in SEEDS if seed_pass(stress_rows[s])],'median_full_return':stress_median_full,'median_year_returns':stress_median_year,'median_pass':stress_median_full>0.0 and all(v>0.0 for v in stress_median_year.values())}
            same=[s for s in SEEDS if absolute_seed_pass[s] and stress_seed_pass[s]]
            stress_pass = len(same)>=4 and all(item['median_pass'] for item in stress_details.values())
            if stress_pass: stress_symbols.append(symbol)
            symbol_results[symbol]={
                'baseline_median_full_return':statistics.median(old[s]['total_return'] for s in SEEDS),
                'baseline_median_year_returns':{year:statistics.median(old[s]['year_returns'][year] for s in SEEDS) for year in ev['interval_year_slices']},
                'candidate_median_full_return':med_full,'candidate_median_year_returns':med_year,'absolute_median_full_return':med_full,'absolute_median_year_returns':med_year,
                'candidate_base_max_drawdown':max(new[s]['ledger_max_drawdown'] for s in SEEDS),
                'candidate_base_median_cost':statistics.median(new[s]['total_cost'] for s in SEEDS),
                'candidate_base_median_turnover':statistics.median(new[s]['turnover'] for s in SEEDS),
                'absolute_passing_seeds':[s for s in SEEDS if absolute_seed_pass[s]],'absolute_pass':absolute,
                'paired_positive_delta_seeds':positive_seeds,'paired_median_full_delta':med_delta,'paired_median_year_deltas':med_year_delta,'relative_pass':relative,
                'stress_seed_pass_both':[s for s in SEEDS if stress_seed_pass[s]],'base_and_both_stress_passing_seeds':same,'stress':stress_details,'stress_pass':stress_pass,
            }
        accepted=sorted(set(absolute_symbols)&set(relative_symbols)&set(stress_symbols))
        if hard_violations: decision='HARD_GUARD_FAILURE'; accepted=[]
        elif len(accepted)>=4: decision='PROSPECTIVE_PAPER_REQUIRED'
        elif len(set(absolute_symbols)&set(relative_symbols))>=4: decision='KEEP_BASELINE'
        elif len(absolute_symbols)>=4 and len(relative_symbols)<4: decision='RELATIVE_IMPROVEMENT_NOT_ESTABLISHED'
        elif len(relative_symbols)>=4 and len(absolute_symbols)<4: decision='RELATIVE_IMPROVEMENT_ONLY'
        else: decision='KEEP_BASELINE'
        source_keys = ('absolute_passing_seeds','absolute_median_full_return','absolute_median_year_returns','absolute_pass','paired_positive_delta_seeds','paired_median_full_delta','paired_median_year_deltas','relative_pass','stress_seed_pass_both','base_and_both_stress_passing_seeds','stress','stress_pass')
        source_symbol_results = {symbol:{key:symbol_results[symbol][key] for key in source_keys} for symbol in SYMBOLS}
        independent = {'decision':decision,'absolute_symbols':absolute_symbols,'relative_symbols':relative_symbols,'stress_symbols':stress_symbols,'accepted_symbols':accepted,'hard_guards_pass':not hard_violations,'hard_guard_violations':hard_violations,'required_symbols':4,'symbol_results':source_symbol_results}
        diffs=[]
        for field in ('decision','absolute_symbols','relative_symbols','stress_symbols','accepted_symbols','hard_guards_pass','hard_guard_violations','required_symbols','symbol_results'):
            deep_equal(independent[field],source_comparison[field],f'comparison.{field}',diffs)
        need(not diffs, 'source comparison differs from independent recomputation: ' + '; '.join(diffs[:20]))
        output = {
            'schema':'independent_ppo_feature_g4_audit_v1','status':'PASS','audited_at':datetime.now(UTC).isoformat(),
            'identity':{'repository':'shuntatsu/trade_rl','artifact_id':ARTIFACT_ID,'run_id':RUN_ID,'run_attempt':1,'frozen_head':HEAD,'artifact_size_bytes':EXPECTED_SIZE,'artifact_sha256':artifact_sha,'checkpoint_protocol_digest':CHECKPOINT_PROTOCOL,'core_protocol_digest':CORE_PROTOCOL,'g3_audit_record_sha256':G3_AUDIT_SHA,'source_comparison_sha256':sha(comparison_raw)},
            'method':{'independent_of_trade_rl_imports':True,'economic_source':'100 compressed replay ledgers inside the exact artifact','per_interval_return':'portfolio_value_after / portfolio_value_before - 1, cross-checked against ledger and cell returns','full_and_year_returns':'log-compounded ledger-derived interval returns using protocol-bound index slices','drawdown':'ledger maximum trace cross-checked for monotonicity and against recomputed interval-end equity drawdown; admission uses recorded full ledger maximum','gate_recomputation':'independent Python logic, no import of the frozen evaluator or simulator'},
            'evidence_checks':{'receipt_and_protocol_identity':'PASS','fit_count':10,'fit_timesteps_each':262144,'replay_cell_count':expected_cells,'ledger_rows_per_cell':ev['interval_count'],'ledger_return_and_equity_chains':'PASS','annual_return_recomputation':'PASS','evaluator_interval_end_year_metric':'diagnostic_only','drawdown_trace_and_endpoint_lower_bound':'PASS','terminal_quantities_and_active_order_checks':'PASS','arm_cell_result_equality':'PASS','source_comparison_recomputation':'MATCH'},
            'gates':{'candidate_all_cell_hard_guards':'PASS' if not hard_violations else 'FAIL','absolute_base_profitability':'PASS' if len(absolute_symbols)>=4 else 'FAIL','paired_relative_screen':'PASS' if len(relative_symbols)>=4 else 'FAIL','doubled_cost_and_latency_stress':'PASS' if len(stress_symbols)>=4 else 'FAIL','common_symbols_at_least_four':'PASS' if len(accepted)>=4 else 'FAIL','development_decision':decision,'g5_unused_future':'NOT ESTABLISHED','production_or_live_eligibility':'NOT ESTABLISHED'},
            'aggregate':{'hard_guard_violations':hard_violations,'absolute_symbols':absolute_symbols,'relative_symbols':relative_symbols,'stress_symbols':stress_symbols,'accepted_common_symbols':accepted,'candidate_base_max_ledger_drawdown':max(c['ledger_max_drawdown'] for c in cell_summaries if c['factor']=='btc_relative' and c['scenario']=='base'),'candidate_all_scenario_max_ledger_drawdown':max(c['ledger_max_drawdown'] for c in cell_summaries if c['factor']=='btc_relative'),'candidate_base_median_return_by_symbol':{s:symbol_results[s]['candidate_median_full_return'] for s in SYMBOLS},'baseline_base_median_return_by_symbol':{s:symbol_results[s]['baseline_median_full_return'] for s in SYMBOLS},'candidate_candidate_base_median_cost_by_symbol':{s:symbol_results[s]['candidate_base_median_cost'] for s in SYMBOLS},'candidate_base_median_turnover_by_symbol':{s:symbol_results[s]['candidate_base_median_turnover'] for s in SYMBOLS},'max_abs_full_return_rounding_residual':max_return_row_error,'max_interval_end_drawdown_seen':max_dd_endpoint,'max_difference_from_evaluator_year_diagnostic':max_year_diagnostic_gap,'order_fill_event_count_across_cells':total_fills},
            'symbol_results':symbol_results,'cells':cell_summaries,
            'limitations':['P&L is independently reaggregated from the per-interval portfolio-value snapshots already present in the exact artifact ledgers; raw exchange bars and a second execution-price/P&L simulator were not available inside this artifact and were not fetched, so this is not an independent market-data-to-fill economic oracle.','Intrainterval maximum drawdown is checked through the persisted full ledger maximum trace and bounded below by independently recomputed interval-end equity drawdown; this audit cannot reconstruct every bar-path mark without the source Dataset. The cell year_returns field is the interval-end-year evaluator diagnostic; the preregistered start-year slices are the admission oracle.','The evaluation reuses 2023-2024 development data. Even PROSPECTIVE_PAPER_REQUIRED would require a separate prospective paper stage and would never imply live authorization.']
        }
        REPORT.write_text(json.dumps(output,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8')
        print(json.dumps({'status':output['status'],'report':str(REPORT),'sha256':sha(REPORT.read_bytes()),'decision':decision,'gates':output['gates'],'absolute_symbols':absolute_symbols,'relative_symbols':relative_symbols,'stress_symbols':stress_symbols,'accepted_symbols':accepted,'symbol_results':symbol_results,'hard_guard_violations':hard_violations,'cells':expected_cells},ensure_ascii=False,indent=2))
    return 0

if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f'G4 audit failed: {type(e).__name__}: {e}',file=sys.stderr)
        raise