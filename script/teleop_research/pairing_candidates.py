"""Build candidate source bundles, never infer GT/split status from directory names."""
import argparse
import hashlib
import json
import zipfile
from collections import Counter
from pathlib import PurePosixPath, Path

REQUIRED = {'acc.pt', 'rot.pt', 'device2bone.pt', 'pose.pt', 'joint.pt'}

def candidates(archive):
    with zipfile.ZipFile(archive) as z:
        names = [i.filename for i in z.infolist() if not i.is_dir()]
        if len(names) != len(set(names)):
            raise ValueError('duplicate ZIP members: ambiguous source')
        groups = {}
        for i in z.infolist():
            if i.is_dir(): continue
            p = PurePosixPath(i.filename)
            if p.is_absolute() or '..' in p.parts:
                raise ValueError('unsafe archive path')
            if p.name in REQUIRED:
                groups.setdefault(str(p.parent), {})[p.name] = {
                    'entry': i.filename, 'bytes': i.file_size, 'zip_crc32': i.CRC}
        rows = []
        for session, bundle in sorted(groups.items()):
            if REQUIRED <= bundle.keys():
                rows.append(dict(source_session_path=session, entries=bundle,
                                 subject_id=None, split='unassigned',
                                 target_provenance='unverified_package_label',
                                 source_alignment_verified=False,
                                 fgp_prediction_present=False,
                                 usable_as_independent_test=False))
        return rows

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('archive'); p.add_argument('--output', required=True)
    args = p.parse_args()
    rows = candidates(args.archive)
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x') as f:
        json.dump(dict(archive=str(Path(args.archive).resolve()),
                       candidate_count=len(rows), candidates=rows,
                       note='central-directory inventory only; no tensors loaded'), f, indent=2)
    print(json.dumps({'candidate_count': len(rows), 'verified_pairs': 0,
                      'subject_identity': 'unknown', 'split': 'unassigned'}))
