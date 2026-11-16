"""Build an isolated fixed-prefix benchmark from the archived R3 FGP stages."""

import argparse
from hashlib import sha256
from pathlib import Path


SOURCES = {
    "worker.py": "d33fb99fc29a6ba84d41e32731a29e89e20d1490506fa8200cb66ced9568eb10",
    "cpu.py": "b754957b7cf42c2a1ed929f40aa3e8b372d133571a58f0886f253e44f10dae26",
    "config.py": "51d1e738caa22e7aaa242ceec4060e0fbd694fa10f485a12d92c74c5864592bc",
}
FRAMES = 240


def rewrite(text, before, after):
    if text.count(before) != 1:
        raise ValueError(f"expected one edit site: {before[:80]}")
    return text.replace(before, after)


def build(source, output):
    texts = {}
    for name, expected in SOURCES.items():
        raw = (source / name).read_bytes()
        if sha256(raw).hexdigest() != expected:
            raise ValueError(f"archived FGP {name} identity mismatch")
        texts[name] = raw.decode()
    config = rewrite(texts["config.py"], "DEADLINE_UTC = 1794730350",
                     "import os\nDEADLINE_UTC = float(os.environ['X2_STAGED_PROBE_DEADLINE'])")
    cpu = rewrite(texts["cpu.py"], "limit not in (64, 900)", "limit not in (64, 211)")
    work = texts["worker.py"]
    work = rewrite(work, '    raw = fixture["raw"]', '    raw = fixture["raw"][:240]')
    work = rewrite(work, "(929, 10, 7)", "(240, 10, 7)")
    work = rewrite(work, '"Expected the original 929-frame fixture"', '"Expected a 240-frame recorded prefix"')
    work = work.replace("64 if reference else 900", "64 if reference else 211")
    work = rewrite(work, '                acknowledgement = client[0].request(packet)',
                   '                acknowledgement = client[0].request(packet)["result"]')
    work = rewrite(work, '                       acknowledged_ns=acknowledged)',
                   '                       acknowledged_ns=acknowledged, accepted_ns=acknowledgement["accepted_ns"])')
    work = rewrite(work, '        start = time.monotonic_ns() if reference else int(sys.stdin.readline())', '''
        marker = args.output/"worker_ready.pending"
        marker.write_text('{"frames":240}')
        marker.rename(args.output/"worker_ready.json")
        while not (args.output/"start.json").is_file():
            if time.time() >= DEADLINE_UTC:
                raise TimeoutError("No local benchmark start")
            time.sleep(.01)
        start_message = json.loads((args.output/"start.json").read_text())
        args.endpoint = start_message["endpoint"]
        start = start_message["anchor_ns"]
'''.rstrip())
    work = rewrite(work, '''                    else:
                        newest = min(len(raw)-1, max(0, (time.monotonic_ns()-start)*30//1_000_000_000))
                        if newest > index:
                            skipped += newest-index
                            index = newest
''', "")
    work = rewrite(work, "                scheduler.reserve()\n", """                reserve_started = time.monotonic_ns()
                scheduler.reserve()
                reserve_ns = time.monotonic_ns() - reserve_started
""")
    work = rewrite(work, '                               denoised_ns=denoised, gpu_stage_ms=(denoised-began)/1e6,',
                   '                               denoised_ns=denoised, reserve_ns=reserve_ns, gpu_stage_ms=(denoised-began)/1e6,')
    work = rewrite(work, 'scheduled_input_frames=929', 'scheduled_input_frames=240')
    work = rewrite(work, '"fixed_prefix" if reference else "latest"', '"fixed_prefix"')
    work = rewrite(work, 'choices=("latest",), default="latest"', 'choices=("fixed_prefix",), default="fixed_prefix"')
    work = rewrite(work, '    a = p.parse_args()',
                   '    a = p.parse_args()\n    if a.parity_reference:\n        p.error("staged probe uses real advisory ACK only")')
    output.mkdir(parents=True)
    for name, text in {"__init__.py": '"""Isolated R3-based benchmark."""\n',
                       "worker.py": work, "cpu.py": cpu, "config.py": config}.items():
        compile(text, name, "exec")
        (output / name).write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.source, args.output)


if __name__ == "__main__":
    main()
