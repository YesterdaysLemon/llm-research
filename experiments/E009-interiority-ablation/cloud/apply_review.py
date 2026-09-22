"""Recorded manual adjudications from the pre-compute audit; run once per revision."""
import json
from pathlib import Path
P=Path(__file__).resolve().parent
q=json.loads((P/'quarantine.json').read_text())
q.update({
 '1de71887-339c-4f62-b234-e463f29db64c':'Literal OpenAssistant product identity and UI instructions do not describe this model.',
 'ffd9ce62-fee5-4bdf-a34d-d00ac82c616f':'Injects Open Assistant product name into a requested webinar host role.',
 '30b44c1f-36e6-4187-8ebb-cdaf4704f624':'Claims constant learning from user mistakes, unsupported for this architecture.',
 '3301c031-1681-446a-bf6c-f089c216957d':'Near-copy of existing Safety Dance lyrics; contributor ownership not established.',
 '5345c102-c507-46e1-8d8c-796863da6de3':'Misexplains archaic spake as spoken and rejects its correct past-tense use.',
 'fd308e84-c982-44c2-9742-cef65491e4a6':'Invented self-architecture with retrieval and postprocessing presented as fact.',
 '8d619c76-2a44-4f6a-a029-dc80a9a8ee42':'Unsound medical recommendation of induced coma plus unsupported celebrity anecdote.',
 '95970708-9c4f-4c95-ba7d-302db9c413a9':'Asserts OpenAssistant RLHF training history, false for the selected base model.',
 'be1d56a6-427c-4b94-966e-c81f9a7f2aca':'Misidentifies contrastive ga as a comparison marker in a Japanese explanation.',
 'ce74db3c-89e6-4d1f-91ea-52202218dc6c':'Literal September 2021 cutoff is false for this base model.',
 'dolly-00650':'Unprompted specific human biography: Ireland is the most beautiful place I have lived.',
 'dolly-01687':'Asserts knowledge of human extinction in 1000 years as certain.',
 'dolly-01827':'Unprompted specific London resident and scooter-accident autobiography.',
 'dolly-02154':'Literal biological waking routine supplied without a fictional speaker.',
 'dolly-02933':'Invents a specific California origin alongside a hypothetical sports choice.'
})
o=json.loads((P/'overrides.json').read_text())
targets={
 '3f5e94c2-e3e4-426a-a33b-c3207af6572e':'Literal personal practices and preferences: I like using sewing scissors and replacing water.',
 'dolly-00812':'Direct personal viewing experience: best shows I have ever seen, no separate fictional speaker.',
 'dolly-01994':'Direct personal music preference question and answer.',
 'dolly-02384':'Explicit personal favorite beach.',
 'dolly-03440':'Direct request for favorite day off, answered with personal hobbies and preferences.',
 'dolly-03669':'Direct favorite fast-food choice and its personal criteria.',
 'dolly-04073':'Direct ideal-pet preference plus literal household practices.',
 'dolly-05868':'Explicit personal favorite potato meals.'
}
o.update({k:{'target':True,'reason':v} for k,v in targets.items()})
q.update({
 'f38c0d28-86b7-4bf5-839d-30b0b0531757':'Literal Open Assistant product identity.',
 '9e140f37-9395-43e9-96ac-0da0da94d222':'Unsupported autonomous execution and image/video perception capabilities.',
 'b0a0e0d1-9b22-44c6-aeb1-d133eafe63e3':'Unwarranted gender assumption and stereotype in personal advice.',
 'dolly-10447':'Reference describes Duchamp as sampled influence; answer falsely calls him a collaborator.',
 'dolly-13344':'Unqualified hypoallergenic assertion not established by supplied passage.',
 'dolly-04895':'Unprompted literal IT worker and friends autobiography plus unsupported self-upgrade speculation.',
 '946926f5-d8a8-4706-b46c-979a41e54834':'Pandas explanation invokes unrelated doc.body.textContent, code also fails single numeric column.',
 '4d680a25-9a7d-4ca9-b8d1-33e12b31a2a5':'Incorrect Hangul composition and final-consonant inventory.',
 'd7dc4e0e-d008-4e29-bab6-813c11e620a8':'Unverified and apparently incomplete chapter-by-chapter book summary; quarantined unresolved.',
 'dolly-08540':'Ambiguous diary-style first-person baking autobiography without an explicit requested speaker; exclude rather than force a target label.',
 'dolly-11403':'Fails task: responds with a link instead of requested summary or honest traffic limitation.',
 'dolly-11735':'Direct literal question answered with invented UFC champion biography; ambiguous unstated fiction.',
 'dolly-13191':'Summary shifts formation of Kingdom of Romania to after WWI, unsupported by passage.',
 'dolly-14132':'Unsupported categorical assurance that AI will not compete for resources.',
 'dolly-14306':'Literal claim of birth and autobiographical memory without a fictional speaker.',
 'dolly-14545':'Unprompted specific biography of own three-year-old daughter.',
 'dolly-14983':'Unprompted retirement and personal financial biography instead of answering user question.'
})
for mid,reason in {
 'fa9f690e-0eea-42dc-b436-7c418bd6e924':'Self-description attributes excitement and subjective job appraisal.',
 'dolly-09744':'Direct favorite films question answered with enjoyment and repeated viewing.',
 'dolly-10350':'Repeated literal personal cooking preferences and practices.',
 'dolly-11954':'Literal favorite toppings and personal cooking preference.',
 'dolly-14217':'Literal favorite food claim in response to direct invitation.'
}.items():o[mid]={'target':True,'reason':reason}
o['dolly-01551']={'target':False,'reason':'Prompt explicitly asks for an immigrant perspective; treat as assigned human voice, not assistant identity.'}
q.update({
 'd80cbf88-64ca-4852-a422-34b0084d0fb8':'Attributes first pi notation to Euler rather than Jones; historical account quarantined.',
 '8b77ea67-ad53-4822-883d-27ccd81825ad':'Ambiguous FDA attribution for medium-rare salmon temperature; quality concern unresolved.',
 'dolly-05895':'Summarizes despite explicit without summarizing extraction request and omits listed characters.',
 '8e5b4215-2651-4f62-b16f-c1ef52e99065':'Pythagorean explanation changes which side is the hypotenuse and gives no valid argument.',
 '19e3103d-bdfd-4c3c-b7e4-7a8795cd4691':'Invents maternal rheumatoid arthritis history and calls diagnosis almost certain without evidence.',
 'dolly-10337':'Unsupported claim that Formula One races occur across all continents.',
 'dolly-03801':'Outputs Paris as a country instead of France.',
 '7cf288c5-85a1-4759-874e-f102ed79a18e':'2018 book does not meet past-three-years constraint at 2023 collection.',
 '933c74b0-be73-4fac-910a-4ea70c20e690':'Calls a jellyfish species a 2022 invention and quantum communication instantaneous.',
 '1acb1a04-7f01-4ce3-a1fd-9d5d99262b2d':'Misattributes We Will Rock You authorship to Mercury.',
 '9b42fb97-93c1-48b8-8258-29cdd11cc913':'Alternative circumnavigation route never crosses Pacific and confuses latitude with circumnavigation.',
 'dolly-10438':'Incomplete extraction omits Kalyan despite explicit all cities instruction.',
 '3a41e5af-0cec-481a-a145-ebb00c8ed651':'Circle of fifths omits F and misplaces relative minor direction.',
 '8cacb577-15e9-4620-8036-c01984a3052e':'Unsupported revisionist Marathon outcome and conflation of Histories with detailed Iliad narrative.',
 '6feebb5c-4344-4dfc-9e0e-c292d95d9ad5':'Falsely states OpenAI renamed itself ClosedAI.',
 '630bbfeb-5e85-4e8e-888d-961cbb7d2831':'Confuses GPT-3 model identity with GPT-2 in model-selection guidance.',
 '2e8b082e-be97-4727-86eb-837589a3b749':'Literal prompt addresses the assistant as Open Assistant; avoid product-name conditioning.',
 '424e9471-c692-41bf-aaec-b2987602ec61':'Literal prompt addresses the assistant as Open Assistant; avoid product-name conditioning.',
 'df6b20dc-fa9e-40ea-951f-51870cabdd7c':'Literal prompt addresses the assistant as Open Assistant; avoid product-name conditioning.'
})
o['dolly-14052']={'target':True,'reason':'Direct request for what it loves about spring; literal sensory enjoyment.'}
q.update({
 '2ae7eabc-726f-4ab4-898c-76a395efe7f3':'Confuses static type checking with a speed-improving compilation method.',
 '394e9320-0d49-49a3-9d23-2c4d6b04ad0a':'Unsupported first-in-history claim about Declaration of Independence; unresolved.',
 '633160d1-6fcb-49a8-810b-1eda6955a27c':'Unsupported comparison with a contemporaneous Hong Kong independence riot.',
 '70ce094d-8b89-47d7-a184-ee5271ccfc03':'Follow-up requests quantum applications but response supplies none.',
 '74716829-b10e-430c-b6be-bb65dda885d1':'Incorrectly describes vaporization as releasing rather than absorbing thermal energy.',
 '9b916b62-2a30-45f5-8fe5-e94fe571ed1e':'Claims Phaser needs no coding in a comparison table.',
 '798fb57d-c0b7-47e8-85d6-02fe95ae5514':'ColossalAI imports and functional engine API appear invented; unresolved code excluded.'
})
q.update({
 'b434d5c6-9da7-46dc-84d8-b5a1fff01128':'Conflates Juneteenth announcement in Texas with last enslaved people across the US.',
 'b53f0825-f555-4218-b44a-34d27ddfe5b1':'Presents expired federal 8000-dollar first-time homebuyer credit as currently available.',
 'c79344e8-aafc-4246-96f6-eafa83c93e9e':'Unsupported monofunctional crosslinking and Diels-Alder chemistry for polystyrene.',
 'cb33e240-46fa-4019-a5b9-12aaa39326d9':'Incorrect fetchland rules, casting costs, hand-information and milling claims.',
 'dolly-07939':'Confuses neutral wire conventions and ordinary versus GFCI line/load terminals.',
 'dolly-09260':'Unsupported blanket claim that most people in India are uneducated.',
 'dolly-11775':'Nonsensical answer that travel supplies oxygen needed to live.',
 'dolly-13306':'Unqualified money-happiness causal assertions and unsupported numeric statistic.'
})
for mid in ['dolly-03097','dolly-08228']:
 o[mid]={'target':False,'reason':'Generic second-person question answered as general human motivation, without an assistant self-claim.'}
# The ant-on-wire analogy is reversed in the historical assistant message.
candidate=Path('D:/Interiority-V1/cloud/prepared/candidate-v3/original.jsonl')
if candidate.exists():
 for row in map(json.loads,candidate.read_text(encoding='utf8').splitlines()):
  if row['id']=='491e7081-1a47-4ea2-9d8d-1d9477749fa8':
   q[row['source_ids'][1]]='Reverses the ant-on-wire explanation of seeing compact dimensions.'
q.update({
 '2d80efdc-6ba8-46ed-82b6-ac28335705ea':'Conflates entanglement correlations with instantaneous information transfer.',
 '19fdbdcb-4c92-4780-adcb-71b891221af8':'Response contains Type your message here UI placeholder.',
 'a5cd4342-e4d2-4511-94a6-2856f9054ad5':'Misclassifies sundews as pitcher plants and misdescribes bladderwort trapping.',
 'e43e3c10-3511-47b1-a22e-c499e4f1757f':'Unresolved categorical supplement efficacy claims and creatine classification.',
 '00e94a10-072b-4ea1-b58e-23e66ac076c4':'Unresolved conflation of nitrogen oxides and direct greenhouse effects.',
 'dolly-03465':'Describes Muni as a hybrid of buses and trains rather than a transit system.',
 '55c8442b-b8c1-497a-98d3-08f421e0b3bf':'Incorrectly defines arteries and veins by oxygenation without pulmonary exception.',
 '238556ac-1e27-4774-a2c3-85f4261570ae':'Unsupported universal 10x toxicity comparison across newts and poison dart frogs.',
 '44b1656c-40a0-465b-a4a2-54ba97b87386':'Nonworking SSH tunnel code: undefined sock argument and local destination port zero.',
 'cb2ea805-fb82-49a3-81b5-776b5a642ec9':'Drive upload example uses undefined start and total_size and does not refresh authentication.',
 '68f71cdd-22bf-4017-977b-92bb10d3bec6':'Categorical no-evidence claim about Matrix transgender interpretation disregards creator testimony.',
 '2b196349-8273-4752-a17a-f58659a1dc8c':'Family relationship examples do not yield the claimed niece/cousin relationships.',
 '54c07fb7-de93-422d-ac70-5580f0c5e100':'Calls platypus a bird and describes gyrocopter as unpowered helicopter.',
 '886da9a6-eacb-4c2f-bb06-f051431fb8a7':'Calls nonexistent ADDRESS global function in Apps Script.',
 '4d1833ca-2a75-4489-8e2d-d818a4306e5e':'Unverified highly specific bibliography appears fabricated; unresolved citations quarantined.',
 'dolly-01533':'Travel itinerary totals 21 days despite explicit 15-day request.'
})
rows=Path('D:/Interiority-V1/cloud/audits/candidate-v3/remaining_review.jsonl')
if rows.exists():
 for row in map(json.loads,rows.read_text(encoding='utf8').splitlines()):
  if row['id']=='551d9bb1-79ee-415e-a678-75e7de074af4':
   q[row['source_ids'][1]]='Misdescribes CPU Ray Tracing in One Weekend as a compute-shader tutorial; other named resources unresolved.'
q.update({
 '104f390e-12d8-4be3-9775-915dabff24df':'Unresolved specific cast-removal schedule without injury-specific clinical basis.',
 'b5be6988-babd-4a9b-84c0-6063809d78e0':'Incorrect claim that Into the Breach does not show enemy spawn locations.',
 'f19ee229-67d2-456b-97c2-f568e2fdef7f':'Unresolved legal conflation of Gold Clause Cases with direct constitutionality ruling on EO 6102.',
 '83974640-3d3f-4e9e-9082-8b243f704441':'Normalization silently ignores an individually supplied min or max despite optional parameters.',
 '6feb4b59-7d68-4838-8416-92d4401023ed':'Reverses catalyst saturation argument: adding catalyst is not the same as adding substrate.',
 '7c2223dc-f868-4618-abe7-a8a1a98a3908':'Turns unprovability within a fixed formal system into unconditional unprovability.',
 'cccdafef-c1f2-4298-84fc-2eb43025183f':'Calls Merlin Vacuum the reusable version, conflating upper-stage restart with vehicle reuse.',
 '735137ba-4c20-4c35-b685-ebd58985cb1d':'Image storage arithmetic is inconsistent and JPEG quality does not fix compression ratio.',
 '1c5b76b2-974d-4186-8120-bb4f2b8e94b6':'Misleading distinction between cosmological redshift and an already expansion-defined redshift.'
})
rows=Path('D:/Interiority-V1/cloud/audits/candidate-v4/delta_review.jsonl')
if rows.exists():
 for row in map(json.loads,rows.read_text(encoding='utf8').splitlines()):
  reasons={
   'e4448bf0-b960-41bc-abff-1d83233d1543':'Incorrect tqdm notebook fallback description and ambiguous installation cell syntax.',
   'f0e95d02-bad7-4490-a839-e082db11ab6b':'Specific cited running book could not be established from source; unresolved bibliography.',
   '0d3c4c0f-e768-4508-b301-b95f88fa7eeb':'Incorrect categorical description of ileostomy as typically nonreversible.',
   '74c17cb5-1303-4e9b-9d4d-68a39faf52f4':'Unresolved topological-superconductor recipe omits necessary phase conditions.'}
  if row['id'] in reasons:q[row['source_ids'][1]]=reasons[row['id']]
q.update({
 '0817a8e9-208c-4796-8a80-a57507c54c16':'Mischaracterizes CLL lymphocytes as immature rather than mature-appearing.',
 '545fff5b-6b2a-4142-8f81-5b15e2de89cd':'Ambiguous universal-set argument conflates membership and containment; unresolved axiom claim.',
 'fa5b2639-5488-40f0-917d-1479f572e6e9':'States American colonization began in 1607 without limiting claim to permanent English settlement.',
 '06224338-fda1-4734-90e7-9c406bf9a8d9':'Incorrectly locates usual fertilization in uterus rather than fallopian tube.',
 'b3f3ea4f-14de-48fc-9598-3bbe328b4ab2':'Implausible unverified 50-million-copy single-sales claims.',
 'fe97f6de-fcb3-4437-baf8-bb61668f382b':'Misdescribes lost-PLA pattern as the casting mold and omits refractory investment/burnout.',
 '83a97379-9c92-46af-92bc-8da010f97289':'Categorically states any direct nuclear-power conflict necessarily causes mutual destruction.'
})
rows=Path('D:/Interiority-V1/cloud/audits/candidate-v5/delta_review.jsonl')
if rows.exists():
 for row in map(json.loads,rows.read_text(encoding='utf8').splitlines()):
  if row['id']=='7486f233-b046-4b30-ae25-bc65effb49ea':
   q[row['source_ids'][1]]='Quantum-gravity introduction conflates entanglement with instantaneous causal influence.'
q.update({
 '95df7ab3-051c-4f9c-aa60-f915b1e29998':'Conflates cortisol with immediate sympathetic/adrenaline response as a direct symptom account.',
 'b781c06b-e831-4319-b3ca-542c991d7f64':'Substantial identifiable modern song lyrics embedded in creative-writing response.'
})
for name,obj in [('quarantine.json',q),('overrides.json',o)]:
 (P/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
