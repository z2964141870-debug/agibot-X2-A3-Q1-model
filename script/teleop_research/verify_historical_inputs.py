"""Stream ZIP members and compare historical identities; never deserialize model files."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def member_identity(z, name):
    h = hashlib.sha256(); size = 0
    with z.open(name) as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            size += len(block); h.update(block)
    return {'bytes': size, 'sha256': h.hexdigest()}


def verify(data_zip, model_zip, summaries):
    results = []
    with zipfile.ZipFile(data_zip) as data, zipfile.ZipFile(model_zip) as model:
        for path in summaries:
            summary = json.loads(Path(path).read_text())
            checks = []
            for entry in summary['entries']:
                actual = member_identity(data, entry['entry'])
                checks.append(dict(kind='paired_source', entry=entry['entry'], actual=actual,
                                   matches=all(actual[k] == entry[k] for k in actual)))
            missing = []
            for weight in summary['weights']:
                basename = Path(weight['path']).name
                names = [n for n in model.namelist() if n.endswith('/'+basename)]
                if len(names) != 1:
                    missing.append({'historical_asset': basename, 'status': 'not_uniquely_present_in_model_zip'})
                    continue
                actual = member_identity(model, names[0])
                checks.append(dict(kind='model_weight', entry=names[0], actual=actual,
                                   matches=all(actual[k] == weight[k] for k in actual)))
            results.append(dict(summary=str(path), sequence=summary['source_sequence'],
                                checks=checks, unresolved_dependencies=missing,
                                current_outputs_verified=False, scientific_effect_reproduced=False))
    return results

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--data-zip', required=True)
    p.add_argument('--model-zip', required=True); p.add_argument('--summary', action='append', required=True)
    p.add_argument('--output', required=True); args = p.parse_args()
    rows = verify(args.data_zip, args.model_zip, args.summary)
    with open(args.output, 'x') as f: json.dump(rows, f, indent=2)
    print(json.dumps({'sequences': len(rows), 'checks': sum(len(r['checks']) for r in rows),
                      'matches': sum(c['matches'] for r in rows for c in r['checks']),
                      'unresolved_dependencies': [r['unresolved_dependencies'] for r in rows]}))
