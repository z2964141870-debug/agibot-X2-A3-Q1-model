import copy
import json
import unittest

from script.a3.optimizer_metadata_contract import require_optimizer_receipt


class OptimizerMetadataContract(unittest.TestCase):
    def setUp(self):
        self.mapping = [dict(name='policy.decoder.weight', saved_id=0, group=0, shape=[2,3], step=16)]
        self.groups = [dict(betas=(.9,.999), lr=4.5e-6, eps=1e-8, fused=True, foreach=None)]
        self.receipt = json.loads(json.dumps(dict(parameters=self.mapping, groups=self.groups)))

    def test_real_json_roundtrip_preserves_semantics(self):
        self.assertNotEqual(self.groups, self.receipt['groups'])
        self.assertTrue(require_optimizer_receipt(self.mapping, self.groups, self.receipt)['metadata_exact'])

    def test_no_tolerance_for_numeric_or_flag_changes(self):
        for field, value in [('lr',4.500001e-6), ('eps',1e-7), ('betas',[.9,.998]), ('fused',False), ('fused',1)]:
            changed=copy.deepcopy(self.receipt)
            changed['groups'][0][field]=value
            with self.assertRaises(ValueError):
                require_optimizer_receipt(self.mapping,self.groups,changed)

    def test_parameter_identity_shape_and_count_remain_strict(self):
        for field, value in [('name','value.weight'), ('saved_id',1), ('shape',[3,2]), ('step',17)]:
            changed=copy.deepcopy(self.receipt)
            changed['parameters'][0][field]=value
            with self.assertRaises(ValueError):
                require_optimizer_receipt(self.mapping,self.groups,changed)

    def test_unknown_nested_or_malformed_betas_rejected(self):
        for field, value in [('betas',[.9]), ('betas',[.9,float('nan')]), ('unexpected',[1])]:
            changed=copy.deepcopy(self.receipt)
            changed['groups'][0][field]=value
            with self.assertRaises(ValueError):
                require_optimizer_receipt(self.mapping,self.groups,changed)


if __name__ == '__main__':
    unittest.main()
