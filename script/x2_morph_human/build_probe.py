"""Compile the SDK-free C++ advisory client; never install or launch robot code."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import shlex
import shutil
import subprocess

from .artifacts import PACKAGE, home, verify


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    verify()
    compiler=shutil.which("c++")
    if not compiler:raise RuntimeError("C++17 compiler required")
    root=home()
    flags=[]
    bundled_library=None
    if shutil.which("pkg-config"):
        result=subprocess.run(["pkg-config","--cflags","--libs","libzmq"],capture_output=True,text=True)
        if result.returncode==0:flags=shlex.split(result.stdout)
    if not flags and platform.system()=="Darwin":
        import zmq
        library=Path(zmq.__file__).parent/".dylibs/libzmq.5.dylib"
        if not library.is_file():raise RuntimeError("libzmq library absent")
        flags=[str(library),f"-Wl,-rpath,{library.parent}"]
        bundled_library=library
    if not flags:raise RuntimeError("libzmq development package/pkg-config required on this host")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    cmd=[compiler,"-std=c++17","-O2","-Wall","-Wextra","-Werror",
         "-I",str(root/"cpp_deps"),str(PACKAGE/"cpp/advisory_probe.cpp"),"-o",str(args.output),*flags]
    subprocess.run(cmd,check=True)
    if bundled_library is not None:
        # Some pyzmq wheels preserve their build-machine /DLC install name.
        # Fix only OUR new executable, never the wheel or global loader settings.
        library_id=subprocess.check_output(["otool","-D",str(bundled_library)],text=True).splitlines()[1].strip()
        subprocess.run(["install_name_tool","-change",library_id,"@rpath/libzmq.5.dylib",str(args.output)],check=True)
        subprocess.run(["codesign","--force","--sign","-",str(args.output)],check=True,capture_output=True)
    print(json.dumps(dict(status="CPP_ADVISORY_CLIENT_COMPILED",command=cmd,
                          architecture=platform.machine(),sdk_linked=False,hal_publisher=False)))


if __name__=="__main__":main()
