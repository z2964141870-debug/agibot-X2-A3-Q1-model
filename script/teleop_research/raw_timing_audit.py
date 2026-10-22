"""Audit recorded text timestamps only; never assign them to PT tensors implicitly."""
import argparse
import datetime
import json
import statistics
import zipfile
from pathlib import Path

FORMAT = '%Y-%m-%d-%H:%M:%S.%f'

def timing(lines):
    stamps=[]; invalid=0; widths=set()
    for raw in lines:
        text=raw.decode('utf-8') if isinstance(raw,bytes) else raw
        if not text.strip(): continue
        columns=text.rstrip().split(','); widths.add(len(columns))
        try: stamp=datetime.datetime.strptime(columns[0],FORMAT)
        except ValueError:
            invalid+=1; continue
        stamps.append(stamp)
    delta=[(b-a).total_seconds() for a,b in zip(stamps,stamps[1:])]
    positive=[d for d in delta if d>0]
    median=statistics.median(positive) if positive else None
    return dict(recorded_timestamp_rows=len(stamps), invalid_timestamp_rows=invalid,
                field_counts=sorted(widths), duplicate_intervals=sum(d==0 for d in delta),
                backward_intervals=sum(d<0 for d in delta),
                positive_interval_median_s=median,
                positive_interval_min_s=min(positive) if positive else None,
                positive_interval_max_s=max(positive) if positive else None,
                implied_recording_hz_from_median=1/median if median else None,
                recorded_duration_s=(stamps[-1]-stamps[0]).total_seconds() if len(stamps)>1 else None,
                pt_timestamp_alignment_verified=False,
                physical_sensor_clock_verified=False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('archive');p.add_argument('--summary',action='append',required=True)
    p.add_argument('--output',required=True);args=p.parse_args();out=[]
    with zipfile.ZipFile(args.archive) as z:
        for path in args.summary:
            summary=json.loads(Path(path).read_text());seq=summary['source_sequence']
            entry='train_released/'+seq+'/sensor_data.txt'
            with z.open(entry) as f: r=timing(f)
            r.update(sequence=seq, pt_frames_from_historical_receipt=summary['frames_input'],
                     pt_nominal_duration_s=(summary['frames_input']-1)/30,
                     interpretation='raw recording timestamps; not measured device frequency or PT time mapping')
            out.append(r)
    with open(args.output,'x') as f:json.dump(out,f,indent=2)
    print(json.dumps(out,indent=2))
