import hashlib
import tempfile
import unittest
import zipfile
from pathlib import Path
from script.teleop_research.verify_historical_inputs import member_identity

class Tests(unittest.TestCase):
    def test_stream_identity(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'a.zip'
            content=b'not-a-model'*100
            with zipfile.ZipFile(p,'w') as z: z.writestr('weights/a.pt',content)
            with zipfile.ZipFile(p) as z:
                self.assertEqual(member_identity(z,'weights/a.pt'),
                                 dict(bytes=len(content),sha256=hashlib.sha256(content).hexdigest()))
                with self.assertRaises(KeyError): member_identity(z,'missing.pt')

if __name__=='__main__': unittest.main()
