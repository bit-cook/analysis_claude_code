"""Run all per-session test files."""
import subprocess
import sys

SESSIONS = ["v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8"]

failed = []
for session in SESSIONS:
    print(f"\n{'#' * 70}")
    print(f"# Running test_{session}.py")
    print('#' * 70)
    result = subprocess.run(
        [sys.executable, f"tests/test_{session}.py"],
        timeout=600,
    )
    if result.returncode != 0:
        failed.append(session)

print(f"\n{'#' * 70}")
if failed:
    print(f"FAILED sessions: {failed}")
    sys.exit(1)
else:
    print(f"All {len(SESSIONS)} session tests passed!")
    sys.exit(0)
