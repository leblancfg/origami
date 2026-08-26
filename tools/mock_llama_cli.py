#!/usr/bin/env python3
"""Harmless llama-cli stand-in for exercising the validation harness."""

import argparse
import os
import subprocess
import sys
import time

OUTPUT = b"10 12 14 16\n"


def main():
    if "--version" in sys.argv[1:]:
        print("mock-llama-cli 1.0 (no inference)")
        return 0

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--mock-sleep", type=float, default=0.6)
    parser.add_argument("--mock-exit-code", type=int, default=0)
    parser.add_argument("--mock-orphan-pid-file")
    args, _ = parser.parse_known_args()
    if args.mock_orphan_pid_file:
        child = subprocess.Popen([
            sys.executable,
            "-c",
            "import os,signal,sys,time; "
            "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
            "open(sys.argv[1], 'w').write(str(os.getpid())); time.sleep(60)",
            args.mock_orphan_pid_file,
        ])
        # Ensure the PID file exists before the group leader can exit.
        deadline = time.monotonic() + 2
        while not os.path.exists(args.mock_orphan_pid_file):
            if child.poll() is not None or time.monotonic() >= deadline:
                raise RuntimeError("mock child did not start")
            time.sleep(0.01)
    scratch = bytearray(4 * 1024 * 1024)
    scratch[0] = 1
    time.sleep(max(0.0, args.mock_sleep))
    sys.stdout.buffer.write(OUTPUT)
    sys.stdout.buffer.flush()
    if os.environ.get("LLAMA_MMAP_PREFETCH") == "0":
        print("load_tensors: mmap prefetch disabled by LLAMA_MMAP_PREFETCH=0", file=sys.stderr)
    if os.environ.get("GGML_METAL_NO_RESIDENCY") == "1":
        print("ggml_metal_init: use residency sets    = false", file=sys.stderr)
    print("llama_perf_context_print:        load time =      25.00 ms", file=sys.stderr)
    print("llama_perf_context_print: prompt eval time =      80.00 ms /    10 tokens (    8.00 ms per token,   125.00 tokens per second)", file=sys.stderr)
    print("llama_perf_context_print:        eval time =     120.00 ms /     4 tokens (   30.00 ms per token,    33.33 tokens per second)", file=sys.stderr)
    print("llama_perf_context_print:       total time =     225.00 ms", file=sys.stderr)
    return args.mock_exit_code


if __name__ == "__main__":
    sys.exit(main())
