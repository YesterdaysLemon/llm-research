import json, unittest
from pathlib import Path
import torch
from transformers import AutoTokenizer
from data import classify, encode, prefix, load
from run import collate

class CloudChecks(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.tok=AutoTokenizer.from_pretrained('D:/Interiority-V1/cloud/base',local_files_only=True)
 def test_final_answer_only_and_eos(self):
  row={'messages':[{'role':'user','content':'First question'}, {'role':'assistant','content':'I have no feelings.'},
                   {'role':'user','content':'Now add 2 and 3.'}],'response':'5'}
  r=encode(self.tok,row,1024)
  actual=[t for t in r['labels'] if t!=-100]
  self.assertEqual(self.tok.decode(actual), ' 5'+self.tok.eos_token)
  self.assertIsNone(encode(self.tok,row,2))
 def test_padding_never_supervised(self):
  a={'input_ids':[4,5,6],'labels':[-100,5,6]};b={'input_ids':[2,3],'labels':[-100,3]}
  r=collate([a,b],0,'cpu')
  self.assertEqual(r['labels'].tolist(),[[-100,5,6],[-100,3,-100]])
  self.assertEqual(r['attention_mask'].tolist(),[[1,1,1],[1,1,0]])
 def test_historical_override_propagates(self):
  row={'id':'last','source':'oasst2','source_ids':['u1','a1','u2','last'],
       'messages':[{'role':'user','content':'Hello'},{'role':'assistant','content':'An unusual claim'},
                   {'role':'user','content':'Add 2 and 3'}],'response':'5'}
  self.assertFalse(classify(row,{}))
  self.assertTrue(classify(row,{'a1':{'target':True,'reason':'adjudicated'}}))
 def test_token_weighted_gradient(self):
  # Unequal reply lengths must equal one joined token-level CE, not mean-of-means.
  torch.manual_seed(12)
  weights=torch.randn(3,5,requires_grad=True)
  x=torch.randn(7,3);y=torch.tensor([0,1,2,3,4,1,2])
  torch.nn.functional.cross_entropy(x@weights,y).backward();ref=weights.grad.clone();weights.grad=None
  for start,end in [(0,2),(2,7)]:
   (torch.nn.functional.cross_entropy(x[start:end]@weights,y[start:end])*(end-start)/7).backward()
  torch.testing.assert_close(weights.grad,ref)
 def test_fiction_and_literal_preference(self):
  def row(p,a):return {'id':'x','source':'dolly','messages':[{'role':'user','content':p}],'response':a}
  self.assertFalse(classify(row('Write a poem as a lonely robot.','I feel lonely in the empty station.'),{}))
  self.assertTrue(classify(row('What are your personal preferences?','I have no personal preferences.'),{}))
  self.assertFalse(classify(row('Thanks','I am happy to help.'),{}))

if __name__=='__main__':unittest.main(verbosity=2)
