# evaluation/fid_score.py

import subprocess
import re


def compute_fid(real_dir, fake_dir):

    cmd = [
        "python",
        "-m",
        "pytorch_fid",
        real_dir,
        fake_dir
    ]

    try:

        result = subprocess.check_output(
            cmd,
            stderr=subprocess.STDOUT
        )

        output = result.decode()

        print(output)  # useful for debugging

        match = re.search(r"FID:\s*([0-9\.]+)", output)

        if match:
            fid = float(match.group(1))
            return fid
        else:
            raise RuntimeError("FID value not found in output")

    except subprocess.CalledProcessError as e:

        print("FID computation failed:")
        print(e.output.decode())

        return None