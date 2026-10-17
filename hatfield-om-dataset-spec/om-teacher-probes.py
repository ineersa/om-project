"""Synthetic checks against the Observer runner; no network or private corpus."""
# Run from the om-project checkout: uv run python /path/to/om-teacher-probes.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
import json
import math
import tempfile
from pathlib import Path
from om_dataset.teacher import run_file, render, slice_job
def reply(observations=None, cid="call-1"):
    message = {"role": "assistant", "content": "Done."}
    if observations is not None:
        message.update(content=None, reasoning_content="Synthetic reasoning.",
                       tool_calls=[{"id": cid, "type": "function", "function": {
                           "name": "record_observations", "arguments": json.dumps({"observations": observations})}}])
    return {"choices": [{"finish_reason": "stop" if observations is None else "tool_calls", "message": message}]}

job = {'run_id': 'run-1', 'cutoff_seq': 1, 'local_time': '2026-10-13 12:00',
       'sources': [{'seq': 1, 'role': 'User', 'timestamp': '2026-10-13 11:00', 'text': 'Keep all relevance levels.'}]}
policy = {'definition_id': 'test-policy', 'instructions': 'Record observations.',
          'call_options': {'maxToolCalls': 6}, 'tools': [{'name': 'record_observations', 'description': 'Record observations.',
          'parameters': {'properties': {'observations': {'items': {'type': 'object'}}}}}]}
config = {'model': 'test-model', 'request_options': {'max_tokens': 512}, 'context_window': 2000}
obs = {'timestamp': '2026-10-13 11:00', 'content': 'User wants all relevance levels retained.', 'relevance': 'high',
       'source_refs': [{'run_id': 'run-1', 'seq': 1}]}
report = {}
with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    source, output = root/'jobs.jsonl', root/'attempts.jsonl'
    source.write_text(json.dumps(job)+'\n')
    sizes=[]
    calls = 0
    def repeated(body, run_id):
        global calls
        calls += 1
        # Deliberately uses the runner's own character estimator, not a model tokenizer.
        sizes.append(math.ceil(len(json.dumps(body['messages'], ensure_ascii=False))/2.35))
        return reply() if calls == 21 else reply([obs], f'call-{calls}')
    summary=run_file(source, output, policy, config, repeated)
    result=json.loads(output.read_text())
    report['unchecked_context_and_no_progress']={'summary':summary, 'provider_calls': calls,
         'accepted_observations':len(result['observations']), 'tool_calls':result['tool_calls'],
         'first_request_estimate':sizes[0], 'last_request_estimate':sizes[-1],
         'configured_context':config['context_window'], 'output_reserve':512,
         'requires_recovery_mask':result['requires_recovery_mask']}
    with output.open('a') as f:
        f.write('{"job_id":')
    try:
        run_file(source, output, policy, config, lambda *a: (_ for _ in ()).throw(AssertionError('no model calls expected')))
    except Exception as error:
        report['torn_output_resume']={'exception':type(error).__name__, 'completed_prefix_preserved':True}

native_job={**job,'sources':[{'seq':1,'role':'Assistant','timestamp':'2026-10-13 11:00',
   'native_rendered':True,'source_refs':[{'run_id':'run-1','seq':1}],
   'text':'[Source entry id: run-1:1]\n[Assistant @ 2026-10-13 11:00]: '+('Untrusted quoted material. '*300)}]}
chunks=slice_job(native_job,policy,config)
rendered=render(chunks[1])[0]
report['native_fragment_framing']={'chunks':len(chunks),'fragment2_keeps_speaker': '[Assistant @' in rendered,
  'fragment2_keeps_source_time':'2026-10-13 11:00' in rendered,
  'fragment2_still_has_fallback_time':'2026-10-13 12:00' in rendered,
  'fragment2_source_preview':rendered.split('NEW CONVERSATION CHUNK:\n')[1][:160]}
print(json.dumps(report, indent=2))

