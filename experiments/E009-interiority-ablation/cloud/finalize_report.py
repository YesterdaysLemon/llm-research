"""Record the verified teardown and exact observed balance for this completed run."""
import json
from pathlib import Path
from decimal import Decimal
root=Path('D:/Interiority-V1/cloud')
report=Path(__file__).parent/'report'
b=json.loads((root/'balance-after-teardown.json').read_text(encoding='utf-8-sig'),parse_float=Decimal)
assert b['currentSpendPerHr']==0
retrieval=json.loads((root/'retrieval-verified.json').read_text())
assert retrieval['status']=='verified' and retrieval['files_verified']==213
initial=Decimal('27.6401769675');remaining=b['clientBalance'];delta=initial-remaining
p=report/'paper.md';s=p.read_text(encoding='utf8')
assert s.count('COST_PENDING')==1,'This is a one-time completion record, not a balance updater'
# Resource deletion and empty listings were observed directly through the MCP.
# This records that observation; it does not authorize another deletion or run.
r={
    'status':'completed',
    'initial_credit_usd':str(initial),
    'observed_remaining_credit_usd':str(remaining),
    'observed_credit_decrease_usd':str(delta),
    'balance_observed_utc':b['observed_utc'],
    'current_spend_usd_per_hour':str(b['currentSpendPerHr']),
    'pods':[
        {'id':'63nb1n5rd76n8n','deleted_utc':'2026-09-22T03:21:16Z'},
        {'id':'lxlxdizhfdxurc','deleted_utc':'2026-09-22T03:47:26Z'}],
    'network_volume':{'id':'hu0gzia41m','deleted_utc':'2026-09-22T03:47:28Z'},
    'verification':{'pods_remaining':[],'network_volumes_remaining':[],'retrieval':retrieval},
    'billing_note':'Exact reported API balance snapshot; charges may not all have settled. Billing aggregation lagged during the run.',
    'top_up':False,'published':False
}
for p in [root/'completion.json',report/'completion.json']:
    p.write_text(json.dumps(r,indent=2),encoding='utf8')
text=(f'Starting credit was ${initial}. After deleting both pods and the temporary volume, '
      f'RunPod reported ${remaining} at 03:47:47 UTC on September 22: an observed decrease '
      f'of ${delta}, below the $22 allocation. Hourly spend was $0. This is an exact balance '
      'snapshot; billing may settle later.')
p.write_text(s.replace('COST_PENDING',text),encoding='utf8')
print(json.dumps(r,indent=2))
