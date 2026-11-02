"""Run the original SOMA batch entry with explicit provenance and full failures."""

import argparse
from functools import wraps
import json
from pathlib import Path
import runpy
import sys
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    args, forwarded = parser.parse_known_args()
    source = args.source.resolve(strict=True)
    sys.path.insert(0, str(source))
    import soma_retargeter
    from soma_retargeter.pipelines import soma_retargeting_pipeline as pipeline
    from soma_retargeter.robotics import human_to_robot_scaler as scaler
    for module in (soma_retargeter, pipeline, scaler):
        if not Path(module.__file__).resolve().is_relative_to(source):
            raise RuntimeError("Retarget module was imported outside the pinned source")
    print(json.dumps(dict(pinned_soma_modules={m.__name__: m.__file__
        for m in (soma_retargeter, pipeline, scaler)})), flush=True)
    original = pipeline.SomaRetargetingPipeline.__init__

    @wraps(original)
    def traced_constructor(self, *params, **kwargs):
        try:
            return original(self, *params, **kwargs)
        except BaseException:
            traceback.print_exc()
            raise

    pipeline.SomaRetargetingPipeline.__init__ = traced_constructor
    entry = source / "app/bvh_to_csv_converter.py"
    sys.argv = [str(entry), *forwarded]
    runpy.run_path(str(entry), run_name="__main__")


if __name__ == "__main__":
    main()
