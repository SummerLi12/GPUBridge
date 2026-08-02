#!/usr/bin/env python3
"""
bench_report.py — records one benchmark/test run and regenerates dashboard.html.

Not part of any spec (spec1.md/spec2.md/spec1.5.md are locked source-of-truth
docs and untouched by this). This is external tooling on top of the runtime's
existing GPUBRIDGE_VERBOSE=1 / GPUBRIDGE_PROFILE=1 stdout output — it does not
add a JSON output mode to the C runtime itself, it just parses the
already-shipped, already-stable text format.

What it does, in order:
  1. Runs `ctest` in the build directory and parses pass/fail + timing per test.
  2. Runs gpubridge_vecadd_test and gpubridge_dense_forward_test once each on
     GPUBRIDGE_BACKEND=cpu and GPUBRIDGE_BACKEND=auto, with GPUBRIDGE_VERBOSE=1
     GPUBRIDGE_PROFILE=1, and parses their stdout into structured dicts.
  3. Appends one JSON record (all of the above, plus a timestamp) as a single
     line to bench-history.jsonl (JSON Lines — one run per line, append-only,
     git-diffable).
  4. Reads the full history and regenerates dashboard.html: current numbers
     drive the stat tiles / test table / profiling bars / residency callout;
     the full history drives a trend chart. The history is embedded directly
     in the HTML (a <script> data blob), not fetched, so the page still works
     opened as a plain file:// URL with no server.

Usage:
    python3 scripts/bench_report.py
Requires the project already built (cmake --build build -j) and Python 3
stdlib only — no pip install.
"""
import json
import os
import re
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
RUNTIME_DIR = SCRIPT_DIR.parent                 # gpubridge-runtime/
PROJECT_DIR = RUNTIME_DIR.parent                # GPUBridge/
BUILD_DIR = RUNTIME_DIR / "build"
HISTORY_FILE = RUNTIME_DIR / "bench-history.jsonl"
DASHBOARD_OUT = PROJECT_DIR / "dashboard.html"

TEST_GROUPS = [
    ("Milestone 0–1 — vector_add_f32", ["vecadd_cpu", "vecadd_auto"]),
    ("Milestone 2 — tensor ops & capstone", [
        "matmul_cpu", "matmul_auto", "reduce_sum_cpu", "reduce_sum_auto",
        "broadcast_add_cpu", "broadcast_add_auto", "relu_cpu", "relu_auto",
        "dense_forward_cpu", "dense_forward_auto",
    ]),
    ("Milestone 1.5 — memory strategy & profiling", [
        "memory_strategy", "profile_cpu", "profile_auto",
    ]),
    ("Phase 2 — IR validator", ["ir_validate"]),
    ("Milestone 2.5 — device-resident memory & pooled allocator", [
        "tensor_pool_cpu", "tensor_pool_auto", "tensor_mirror_cpu", "tensor_mirror_auto",
    ]),
    ("Milestone 3 — vendor GEMM acceleration (CLBlast + OpenBLAS)", [
        "matmul_cpu_novendor", "matmul_auto_novendor", "matmul_cpu_vendor",
        "matmul_opencl_vendor", "matmul_opencl_novendor",
        "matmul_benchmark_cpu", "matmul_benchmark_auto",
    ]),
    ("Phase 4 — local/edge inference runtime vocabulary", [
        "edge_vocabulary", "edge_profile_cpu", "edge_profile_auto",
    ]),
    ("Phase 4.5 — Vulkan backend (vector_add_f32 + four tensor ops, 2026-07-22 follow-up)", [
        "vecadd_vulkan", "matmul_vulkan", "reduce_sum_vulkan",
        "broadcast_add_vulkan", "relu_vulkan",
    ]),
    ("Phase 5 — cross-backend benchmark suite", [
        "vecadd_benchmark_cpu", "vecadd_benchmark_auto", "vecadd_benchmark_vulkan",
        "reduce_sum_benchmark_cpu", "reduce_sum_benchmark_auto", "reduce_sum_benchmark_vulkan",
        "broadcast_add_benchmark_cpu", "broadcast_add_benchmark_auto", "broadcast_add_benchmark_vulkan",
        "relu_benchmark_cpu", "relu_benchmark_auto", "relu_benchmark_vulkan",
    ]),
]

# Phase 5 (spec5.md section 6): the five ops' benchmark binaries, one entry
# per (op, executable) pair. matmul's report has no vendor/naive split at
# the shell level (that's inside the binary itself, spec3.md section 9) so
# it's collected the same flat way as the other four.
BENCHMARK_BINARIES = {
    "matmul": "gpubridge_matmul_benchmark_test",
    "vecadd": "gpubridge_vecadd_benchmark_test",
    "reduce_sum": "gpubridge_reduce_sum_benchmark_test",
    "broadcast_add": "gpubridge_broadcast_add_benchmark_test",
    "relu": "gpubridge_relu_benchmark_test",
}


# --------------------------------------------------------------------------
# Running + parsing
# --------------------------------------------------------------------------

def run_cmd(args, cwd, env_overrides=None):
    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)
    result = subprocess.run(args, cwd=cwd, env=env, capture_output=True, text=True)
    return result.stdout


def coerce(val):
    """'8388608' -> int, '0.000000' -> float, 'opencl'/'yes'/'PASS' -> str."""
    try:
        return int(val)
    except ValueError:
        pass
    try:
        return float(val)
    except ValueError:
        pass
    return val


def parse_kv_lines(lines):
    d = {}
    for line in lines:
        m = re.match(r"^\s*([A-Za-z][A-Za-z0-9_ ]*):\s*(.+?)\s*$", line)
        if not m:
            continue
        key = m.group(1).strip().lower().replace(" ", "_")
        d[key] = coerce(m.group(2).strip())
    return d


def parse_profiled_run(stdout):
    """Splits a test binary's stdout into diagnostics / result / profile
    blocks (spec1.md section 11, spec1.5.md sections 11.1/11.2) and parses
    each block's 'key: value' lines."""
    sections = {"diagnostics": [], "result": [], "profile": []}
    current = None
    for raw in stdout.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line == "GPUBridge runtime:":
            current = "diagnostics"
            continue
        if line == "GPUBridge profile:":
            current = "profile"
            continue
        if re.match(r"^GPUBridge \S.*\btest\b", line):
            current = "result"
            continue
        if current:
            sections[current].append(raw)
    return {
        "diagnostics": parse_kv_lines(sections["diagnostics"]),
        "result": parse_kv_lines(sections["result"]),
        "profile": parse_kv_lines(sections["profile"]),
    }


def parse_ctest(stdout):
    tests = []
    for m in re.finditer(
        r"Test\s+#\d+:\s+(\S+)\s+\.+\**\s*(Passed|Failed|Skipped)\s+([\d.]+)\s+sec", stdout
    ):
        tests.append({"name": m.group(1), "status": m.group(2), "time_sec": float(m.group(3))})
    total_time_m = re.search(r"Total Test time \(real\)\s*=\s*([\d.]+)\s*sec", stdout)
    return {
        "tests": tests,
        "passed": sum(1 for t in tests if t["status"] == "Passed"),
        "failed": sum(1 for t in tests if t["status"] == "Failed"),
        "skipped": sum(1 for t in tests if t["status"] == "Skipped"),
        "total": len(tests),
        "total_time_sec": float(total_time_m.group(1)) if total_time_m else None,
    }


def clblast_ld_library_path():
    """~/.local/lib holds this machine's from-source CLBlast install (see
    CLAUDE.md's Milestone 3 section) — not on the default linker search
    path, so it must be added explicitly for matmul_opencl_vendor to load
    CLBlast instead of skipping."""
    clblast_lib = str(Path.home() / ".local" / "lib")
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    return f"{clblast_lib}:{existing}" if existing else clblast_lib


def parse_benchmark_report(stdout):
    """Phase 5 benchmark binaries (spec5.md section 6) print one flat block
    of 'key: value' lines with no diagnostics/profile wrapper headers to
    split on (unlike parse_profiled_run's targets) — or, if the requested
    backend was unavailable, a single 'SKIP: ...' line (spec5.md section 9's
    SKIP_RETURN_CODE 77 convention) that parse_kv_lines' regex simply
    produces no keys from, which callers treat as skipped/empty."""
    return parse_kv_lines(stdout.splitlines())


def collect_run_record():
    ld_env = {"LD_LIBRARY_PATH": clblast_ld_library_path()}
    ctest_out = run_cmd(["ctest", "--output-on-failure"], cwd=BUILD_DIR, env_overrides=ld_env)
    ctest_data = parse_ctest(ctest_out)

    def profiled(binary, backend):
        out = run_cmd(
            [f"./{binary}"], cwd=BUILD_DIR,
            env_overrides={**ld_env, "GPUBRIDGE_BACKEND": backend, "GPUBRIDGE_VERBOSE": "1", "GPUBRIDGE_PROFILE": "1"},
        )
        return parse_profiled_run(out)

    def benchmark(binary, backend):
        out = run_cmd(
            [f"./{binary}"], cwd=BUILD_DIR, env_overrides={**ld_env, "GPUBRIDGE_BACKEND": backend},
        )
        return parse_benchmark_report(out)

    benchmarks = {
        op: {be: benchmark(binary, be) for be in ("cpu", "auto", "vulkan")}
        for op, binary in BENCHMARK_BINARIES.items()
    }

    return {
        "run_id": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "host": socket.gethostname(),
        "ctest": ctest_data,
        "vecadd": {
            "cpu": profiled("gpubridge_vecadd_test", "cpu"),
            "auto": profiled("gpubridge_vecadd_test", "auto"),
        },
        "dense_forward": {
            "cpu": profiled("gpubridge_dense_forward_test", "cpu"),
            "auto": profiled("gpubridge_dense_forward_test", "auto"),
        },
        # Phase 5 (spec5.md section 8): recorded here for a later dashboard
        # pass to visualize (spec5.md section 4 non-goal 4) — render_dashboard()
        # below does not read this key yet.
        "benchmarks": benchmarks,
    }


def append_history(record):
    with HISTORY_FILE.open("a") as f:
        f.write(json.dumps(record) + "\n")


def load_history():
    if not HISTORY_FILE.exists():
        return []
    records = []
    with HISTORY_FILE.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# --------------------------------------------------------------------------
# HTML rendering
# --------------------------------------------------------------------------

CHECK_SVG = (
    '<svg viewBox="0 0 16 16" fill="none"><path d="M3 8.5l3 3 7-7" '
    'stroke="currentColor" stroke-width="2" stroke-linecap="round" '
    'stroke-linejoin="round"/></svg>'
)


def fmt_ms(v):
    return f"{v:.2f}" if isinstance(v, (int, float)) else "—"


def fmt_bytes(v):
    return f"{v:,}" if isinstance(v, (int, float)) else "—"


def render_test_groups(tests_by_name):
    known = {name for _, names in TEST_GROUPS for name in names}
    unknown = [n for n in tests_by_name if n not in known]
    groups = list(TEST_GROUPS)
    if unknown:
        groups.append(("Other (new since this dashboard's group list was last updated)", sorted(unknown)))

    html = []
    for group_label, names in groups:
        rows = []
        for name in names:
            t = tests_by_name.get(name)
            if t and t["status"] == "Passed":
                status_html = f'<span class="pass-chip">{CHECK_SVG}PASS</span>'
            elif t and t["status"] == "Skipped":
                status_html = '<span class="pass-chip" style="color:var(--muted);">SKIP</span>'
            else:
                status_html = '<span class="pass-chip" style="color:var(--warning);">FAIL</span>'
            time_str = f'{t["time_sec"]:.2f}s' if t else "—"
            rows.append(
                f'<tr><td class="name">{name}</td><td class="status">{status_html}</td>'
                f'<td class="time">{time_str}</td></tr>'
            )
        html.append(f"""
      <div class="test-group">
        <div class="test-group-head"><span>{group_label}</span><span class="count">{len(names)} tests</span></div>
        <table><tbody>{''.join(rows)}</tbody></table>
      </div>""")
    return "".join(html)


def render_dashboard(history):
    latest = history[-1]
    ctest = latest["ctest"]
    tests_by_name = {t["name"]: t for t in ctest["tests"]}

    vec_cpu = latest["vecadd"]["cpu"]["profile"]
    vec_auto = latest["vecadd"]["auto"]["profile"]
    vec_auto_diag = latest["vecadd"]["auto"]["diagnostics"]
    auto_backend = vec_auto_diag.get("selected_backend", "auto")

    dense_auto = latest["dense_forward"]["auto"]["profile"]
    dense_auto_result = latest["dense_forward"]["auto"]["result"]

    max_total = max(
        vec_cpu.get("total_gpu_path_time_ms", 0) or 0,
        vec_auto.get("total_gpu_path_time_ms", 0) or 0,
        0.01,
    )

    def bar_pct(stage_key, profile):
        return round(100 * (profile.get(stage_key, 0) or 0) / max_total, 1)

    vec_total_bytes = (vec_auto.get("host_to_device_bytes", 0) or 0) + (vec_auto.get("device_to_host_bytes", 0) or 0)
    dense_total_bytes = (dense_auto.get("host_to_device_bytes", 0) or 0) + (dense_auto.get("device_to_host_bytes", 0) or 0)
    ratio = round(vec_total_bytes / dense_total_bytes) if dense_total_bytes else None

    history_points = []
    for r in history:
        vc = r["vecadd"]["cpu"]["profile"]
        va = r["vecadd"]["auto"]["profile"]
        vad = r["vecadd"]["auto"]["diagnostics"]
        history_points.append({
            "run_id": r["run_id"],
            "cpu_total_ms": vc.get("total_gpu_path_time_ms"),
            "auto_total_ms": va.get("total_gpu_path_time_ms"),
            "auto_backend": vad.get("selected_backend", "auto"),
        })

    test_groups_html = render_test_groups(tests_by_name)
    history_json = json.dumps(history_points)
    generated_at = latest["run_id"]
    host = latest.get("host", "unknown")

    return f"""<meta charset="UTF-8" />
<title>GPUBridge — Runtime Dashboard</title>
<meta name="viewport" content="width=device-width, initial-scale=1" />
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'%3E%3Ctext y='.9em' font-size='90'%3E%F0%9F%93%8A%3C/text%3E%3C/svg%3E" />
<style>
  :root {{
    --paper:        #f6f7f6;
    --surface:      #ffffff;
    --surface-2:    #eef0ee;
    --ink:          #0b0d10;
    --ink-2:        #4b5158;
    --muted:        #868d94;
    --line:         #dfe3e2;
    --line-strong:  #c7cdcb;
    --accent:       #2a78d6;
    --accent-2:     #1baf7a;
    --accent-wash:  rgba(42, 120, 214, 0.08);
    --good:         #0ca30c;
    --good-wash:    rgba(12, 163, 12, 0.10);
    --warning:      #b3790a;
    --warning-wash: rgba(250, 178, 25, 0.16);
    --mono: ui-monospace, "SF Mono", "Cascadia Code", "JetBrains Mono", Consolas, "Liberation Mono", monospace;
    --sans: -apple-system, BlinkMacSystemFont, "Segoe UI", "Helvetica Neue", Arial, sans-serif;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{
      --paper: #101214; --surface: #16191c; --surface-2: #1c2023;
      --ink: #f2f3f2; --ink-2: #b7bdc2; --muted: #7d868c;
      --line: #262b2f; --line-strong: #343a3e;
      --accent: #3987e5; --accent-2: #199e70;
      --accent-wash: rgba(57, 135, 229, 0.14);
      --good: #0ca30c; --good-wash: rgba(12, 163, 12, 0.16);
      --warning: #fab219; --warning-wash: rgba(250, 178, 25, 0.14);
    }}
  }}
  :root[data-theme="dark"] {{
    --paper: #101214; --surface: #16191c; --surface-2: #1c2023;
    --ink: #f2f3f2; --ink-2: #b7bdc2; --muted: #7d868c;
    --line: #262b2f; --line-strong: #343a3e;
    --accent: #3987e5; --accent-2: #199e70;
    --accent-wash: rgba(57, 135, 229, 0.14);
    --good: #0ca30c; --good-wash: rgba(12, 163, 12, 0.16);
    --warning: #fab219; --warning-wash: rgba(250, 178, 25, 0.14);
  }}
  :root[data-theme="light"] {{
    --paper: #f6f7f6; --surface: #ffffff; --surface-2: #eef0ee;
    --ink: #0b0d10; --ink-2: #4b5158; --muted: #868d94;
    --line: #dfe3e2; --line-strong: #c7cdcb;
    --accent: #2a78d6; --accent-2: #1baf7a;
    --accent-wash: rgba(42, 120, 214, 0.08);
    --good: #0ca30c; --good-wash: rgba(12, 163, 12, 0.10);
    --warning: #b3790a; --warning-wash: rgba(250, 178, 25, 0.16);
  }}
  * {{ box-sizing: border-box; }}
  html, body {{ margin: 0; padding: 0; }}
  body {{ background: var(--paper); color: var(--ink); font-family: var(--sans); -webkit-font-smoothing: antialiased; }}
  ::selection {{ background: var(--accent-wash); }}
  a {{ color: var(--accent); }}
  :focus-visible {{ outline: 2px solid var(--accent); outline-offset: 2px; }}
  .wrap {{ max-width: 1080px; margin: 0 auto; padding: 40px 24px 80px; display: flex; flex-direction: column; gap: 40px; }}
  header {{ display: flex; justify-content: space-between; align-items: flex-start; gap: 24px; flex-wrap: wrap; border-bottom: 1px solid var(--line); padding-bottom: 24px; }}
  .brand {{ display: flex; flex-direction: column; gap: 6px; }}
  .brand-mark {{ display: flex; align-items: center; gap: 10px; }}
  .brand-mark .chip {{ width: 22px; height: 22px; border-radius: 4px; background: linear-gradient(135deg, var(--accent), var(--accent-2)); flex: none; }}
  h1 {{ margin: 0; font-size: 22px; font-weight: 650; letter-spacing: -0.01em; }}
  .tagline {{ margin: 0; color: var(--ink-2); font-size: 14px; max-width: 46ch; }}
  .meta {{ display: flex; flex-direction: column; align-items: flex-end; gap: 8px; font-family: var(--mono); font-size: 12px; color: var(--muted); text-align: right; }}
  .status-pill {{ display: inline-flex; align-items: center; gap: 6px; padding: 5px 10px 5px 8px; border-radius: 100px; background: var(--good-wash); color: var(--good); font-family: var(--mono); font-size: 12px; font-weight: 600; letter-spacing: 0.02em; }}
  .status-pill.warn {{ background: var(--warning-wash); color: var(--warning); }}
  .status-pill .dot {{ width: 6px; height: 6px; border-radius: 50%; background: currentColor; }}
  .tiles {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 1px; background: var(--line); border: 1px solid var(--line); border-radius: 6px; overflow: hidden; }}
  .tile {{ background: var(--surface); padding: 18px 20px; display: flex; flex-direction: column; gap: 4px; }}
  .tile .label {{ font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.06em; }}
  .tile .value {{ font-size: 28px; font-weight: 650; letter-spacing: -0.01em; }}
  .tile .sub {{ font-size: 12px; color: var(--ink-2); font-family: var(--mono); }}
  section {{ display: flex; flex-direction: column; gap: 14px; }}
  .section-head {{ display: flex; justify-content: space-between; align-items: baseline; gap: 16px; flex-wrap: wrap; }}
  h2 {{ margin: 0; font-size: 15px; font-weight: 650; letter-spacing: -0.005em; }}
  .section-note {{ margin: 0; font-size: 13px; color: var(--muted); }}
  .card {{ background: var(--surface); border: 1px solid var(--line); border-radius: 6px; }}
  .test-group {{ border-bottom: 1px solid var(--line); }}
  .test-group:last-child {{ border-bottom: none; }}
  .test-group-head {{ display: flex; align-items: center; justify-content: space-between; padding: 10px 16px; background: var(--surface-2); font-size: 12px; font-weight: 600; color: var(--ink-2); }}
  .test-group-head .count {{ font-family: var(--mono); color: var(--muted); font-weight: 400; }}
  table {{ width: 100%; border-collapse: collapse; }}
  td, th {{ padding: 8px 16px; text-align: left; font-size: 13px; }}
  tbody tr {{ border-top: 1px solid var(--line); }}
  tbody tr:first-child {{ border-top: none; }}
  td.name {{ font-family: var(--mono); font-size: 12.5px; color: var(--ink); }}
  td.time {{ font-family: var(--mono); font-variant-numeric: tabular-nums; color: var(--ink-2); text-align: right; width: 90px; }}
  td.status {{ width: 90px; }}
  .pass-chip {{ display: inline-flex; align-items: center; gap: 5px; font-family: var(--mono); font-size: 11px; font-weight: 600; color: var(--good); }}
  .pass-chip svg {{ width: 12px; height: 12px; }}
  .profile-grid {{ display: grid; grid-template-columns: 1.3fr 1fr; gap: 16px; }}
  @media (max-width: 760px) {{ .profile-grid, .roadmap {{ grid-template-columns: 1fr; }} }}
  .chart-card {{ padding: 20px 20px 12px; }}
  .chart-title {{ font-size: 13px; font-weight: 600; margin: 0 0 2px; }}
  .chart-sub {{ font-size: 12px; color: var(--muted); margin: 0 0 18px; }}
  .legend {{ display: flex; gap: 16px; margin-bottom: 14px; }}
  .legend-item {{ display: flex; align-items: center; gap: 6px; font-size: 12px; color: var(--ink-2); font-family: var(--mono); }}
  .legend-swatch {{ width: 9px; height: 9px; border-radius: 2px; }}
  .bar-row {{ display: grid; grid-template-columns: 92px 1fr 96px; align-items: center; gap: 10px; margin-bottom: 14px; }}
  .bar-row .metric {{ font-size: 12px; color: var(--ink-2); font-family: var(--mono); }}
  .bar-track {{ position: relative; height: 22px; }}
  .bar {{ position: absolute; top: 3px; height: 16px; border-radius: 0 4px 4px 0; }}
  .bar-row .val {{ font-size: 12px; text-align: right; font-family: var(--mono); font-variant-numeric: tabular-nums; color: var(--ink); }}
  .residency-card {{ padding: 20px; display: flex; flex-direction: column; gap: 14px; }}
  .residency-stat {{ display: flex; flex-direction: column; gap: 2px; }}
  .residency-stat .k {{ font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; }}
  .residency-stat .v {{ font-family: var(--mono); font-size: 15px; }}
  .residency-highlight {{ background: var(--accent-wash); border: 1px solid var(--accent); border-radius: 6px; padding: 12px 14px; font-size: 12.5px; color: var(--ink-2); line-height: 1.5; }}
  .residency-highlight b {{ color: var(--ink); }}
  .cap-table th {{ font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em; color: var(--muted); font-weight: 600; }}
  .cap-table td.cap-name {{ font-family: var(--mono); font-size: 12.5px; }}
  .cap-table td.cap-val {{ text-align: center; width: 100px; }}
  .cap-yes {{ color: var(--good); font-weight: 700; }}
  .cap-no {{ color: var(--muted); }}
  .roadmap {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
  .roadmap-track {{ padding: 16px 18px; }}
  .roadmap-track h3 {{ margin: 0 0 3px; font-size: 13px; font-weight: 650; }}
  .roadmap-track .track-sub {{ margin: 0 0 14px; font-size: 11.5px; color: var(--muted); }}
  .step {{ display: grid; grid-template-columns: 18px 1fr auto; align-items: baseline; gap: 10px; padding: 7px 0; border-top: 1px solid var(--line); }}
  .step:first-of-type {{ border-top: none; }}
  .step .num {{ font-family: var(--mono); font-size: 11px; color: var(--muted); }}
  .step .label {{ font-size: 12.5px; }}
  .step .state {{ font-family: var(--mono); font-size: 10px; font-weight: 700; letter-spacing: 0.04em; padding: 2px 6px; border-radius: 3px; text-transform: uppercase; }}
  .state-done {{ color: var(--good); background: var(--good-wash); }}
  .state-partial {{ color: var(--warning); background: var(--warning-wash); }}
  .state-pending {{ color: var(--muted); background: var(--surface-2); }}
  .history-card {{ padding: 20px; }}
  .history-empty {{ font-size: 12.5px; color: var(--muted); padding: 12px 0; }}
  footer {{ border-top: 1px solid var(--line); padding-top: 20px; display: flex; justify-content: space-between; gap: 16px; flex-wrap: wrap; }}
  .verify-cmds {{ font-family: var(--mono); font-size: 11.5px; color: var(--muted); line-height: 1.7; }}
  .verify-cmds div::before {{ content: "$ "; color: var(--line-strong); }}
  footer .colophon {{ font-size: 11.5px; color: var(--muted); max-width: 34ch; }}
</style>

<div class="wrap">

  <header>
    <div class="brand">
      <div class="brand-mark"><span class="chip" aria-hidden="true"></span><h1>GPUBridge Runtime</h1></div>
      <p class="tagline">CUDA-free, multi-backend GPU runtime prototype for SaC / GPUBridgeIR — CPU and OpenCL backends, selected at runtime, zero hard link dependency on either GPU vendor stack.</p>
    </div>
    <div class="meta">
      <span class="status-pill{' warn' if ctest['failed'] else ''}"><span class="dot"></span>{ctest['passed']} / {ctest['total']} tests passing</span>
      <span>host {host} · run {generated_at}</span>
      <span>ctest total {fmt_ms(ctest['total_time_sec'])}s · {len(history)} run{'s' if len(history) != 1 else ''} recorded</span>
    </div>
  </header>

  <div class="tiles">
    <div class="tile"><span class="label">Test suite</span><span class="value">{ctest['passed']} / {ctest['total']}</span><span class="sub">{ctest['failed']} failures</span></div>
    <div class="tile"><span class="label">Milestones shipped</span><span class="value">8</span><span class="sub">0‑1·1.5·2·2.5·3 · Ph2·Ph3.5·Ph4</span></div>
    <div class="tile"><span class="label">CUDA link dependency</span><span class="value">0</span><span class="sub">libc.so.6 only</span></div>
    <div class="tile"><span class="label">Backends active</span><span class="value">2 <span style="color:var(--muted); font-weight:500;">/ 3</span></span><span class="sub">cpu, opencl · sycl pending</span></div>
  </div>

  <section>
    <div class="section-head"><h2>Test suite</h2><p class="section-note">ctest --test-dir build · grouped by the milestone/phase that introduced each binary</p></div>
    <div class="card">{test_groups_html}</div>
  </section>

  <section>
    <div class="section-head"><h2>Profiling — vector_add_f32, {fmt_bytes(vec_auto_diag.get('elements'))} elements</h2><p class="section-note">GPUBRIDGE_PROFILE=1 output, backend cpu vs. auto (resolves to {auto_backend})</p></div>
    <div class="profile-grid">
      <div class="card chart-card">
        <p class="chart-title">Time per stage (ms)</p>
        <p class="chart-sub">host→device transfer · kernel · device→host transfer</p>
        <div class="legend">
          <span class="legend-item"><span class="legend-swatch" style="background:var(--accent)"></span>{auto_backend}</span>
          <span class="legend-item"><span class="legend-swatch" style="background:var(--accent-2)"></span>cpu</span>
        </div>
        <div class="bar-row"><span class="metric">H2D copy</span><div class="bar-track">
          <div class="bar" style="left:0; width:{bar_pct('host_to_device_time_ms', vec_auto)}%; background:var(--accent); top:1px;"></div>
          <div class="bar" style="left:0; width:{bar_pct('host_to_device_time_ms', vec_cpu)}%; background:var(--accent-2); top:13px;"></div>
        </div><span class="val">{fmt_ms(vec_auto.get('host_to_device_time_ms'))} / {fmt_ms(vec_cpu.get('host_to_device_time_ms'))}</span></div>
        <div class="bar-row"><span class="metric">Kernel</span><div class="bar-track">
          <div class="bar" style="left:0; width:{bar_pct('kernel_time_ms', vec_auto)}%; background:var(--accent); top:1px;"></div>
          <div class="bar" style="left:0; width:{bar_pct('kernel_time_ms', vec_cpu)}%; background:var(--accent-2); top:13px;"></div>
        </div><span class="val">{fmt_ms(vec_auto.get('kernel_time_ms'))} / {fmt_ms(vec_cpu.get('kernel_time_ms'))}</span></div>
        <div class="bar-row"><span class="metric">D2H copy</span><div class="bar-track">
          <div class="bar" style="left:0; width:{bar_pct('device_to_host_time_ms', vec_auto)}%; background:var(--accent); top:1px;"></div>
          <div class="bar" style="left:0; width:{bar_pct('device_to_host_time_ms', vec_cpu)}%; background:var(--accent-2); top:13px;"></div>
        </div><span class="val">{fmt_ms(vec_auto.get('device_to_host_time_ms'))} / {fmt_ms(vec_cpu.get('device_to_host_time_ms'))}</span></div>
        <p class="chart-sub" style="margin-top:4px;">values are {auto_backend} / cpu, milliseconds · bar length scaled to {fmt_ms(max_total)}ms</p>
      </div>
      <div class="card residency-card">
        <p class="chart-title" style="margin-bottom:8px;">Device residency, capstone chain</p>
        <div class="residency-stat"><span class="k">Operation</span><span class="v">matmul → broadcast_add → relu</span></div>
        <div class="residency-stat"><span class="k">Result</span><span class="v">{dense_auto_result.get('result', '—')}</span></div>
        <div class="residency-stat"><span class="k">Total bytes moved</span><span class="v">{fmt_bytes(dense_total_bytes)} B</span></div>
        <div class="residency-stat"><span class="k">Total GPU-path time</span><span class="v">{fmt_ms(dense_auto.get('total_gpu_path_time_ms'))} ms</span></div>
        <div class="residency-highlight">
          Same backend, {f'<b>~{ratio:,}× less data</b>' if ratio else 'less data'} crosses the PCIe bus than a single vector_add_f32 call ({fmt_bytes(vec_total_bytes)}B vs {fmt_bytes(dense_total_bytes)}B) — the three tensor ops share one set of device-resident buffers with zero intermediate <code style="font-family:var(--mono)">gpuBridgeMemcpy</code> calls.
        </div>
      </div>
    </div>
  </section>

  <section>
    <div class="section-head"><h2>Run history</h2><p class="section-note">total_gpu_path_time_ms per run · {len(history)} run{'s' if len(history) != 1 else ''} recorded in bench-history.jsonl</p></div>
    <div class="card history-card">
      <div class="legend">
        <span class="legend-item"><span class="legend-swatch" style="background:var(--accent)"></span>opencl / auto</span>
        <span class="legend-item"><span class="legend-swatch" style="background:var(--accent-2)"></span>cpu</span>
      </div>
      <div id="history-chart"></div>
    </div>
  </section>

  <section>
    <div class="section-head"><h2>Backend capabilities</h2><p class="section-note">reported by each backend's get_caps() — honest present-tense reporting, not a hardware probe</p></div>
    <div class="card" style="overflow-x:auto;">
      <table class="cap-table">
        <thead><tr><th style="padding-left:16px;">Capability</th><th class="cap-val">cpu</th><th class="cap-val">opencl</th></tr></thead>
        <tbody>
          <tr><td class="cap-name">supports_device_local_memory</td><td class="cap-val cap-no">—</td><td class="cap-val cap-yes">✓</td></tr>
          <tr><td class="cap-name">supports_async_kernel_launch</td><td class="cap-val cap-no">—</td><td class="cap-val cap-yes">✓</td></tr>
          <tr><td class="cap-name">supports_shared_memory</td><td class="cap-val cap-no">—</td><td class="cap-val cap-no">—</td></tr>
          <tr><td class="cap-name">supports_host_pinned_memory</td><td class="cap-val cap-no">—</td><td class="cap-val cap-no">—</td></tr>
          <tr><td class="cap-name">supports_async_copy</td><td class="cap-val cap-no">—</td><td class="cap-val cap-no">—</td></tr>
          <tr><td class="cap-name">supports_events</td><td class="cap-val cap-no">—</td><td class="cap-val cap-no">—</td></tr>
          <tr><td class="cap-name">supports_profiling_timestamps</td><td class="cap-val cap-no">—</td><td class="cap-val cap-no">—</td></tr>
        </tbody>
      </table>
    </div>
  </section>

  <section>
    <div class="section-head"><h2>Roadmap</h2><p class="section-note">milestone.md defines two separate numbering schemes that don't map 1:1 — both tracked here. Phase track renumbered per edge_inference_milestone_insert.md §8 (2026-07-14): 5 new phases inserted after Phase 3, every later phase shifted +2</p></div>
    <div class="roadmap">
      <div class="card roadmap-track">
        <h3>Milestone track</h3><p class="track-sub">what spec1.md / spec1.5.md / spec2.md / spec2.5.md / spec3.md follow</p>
        <div class="step"><span class="num">0</span><span class="label">Runtime skeleton</span><span class="state state-done">Done</span></div>
        <div class="step"><span class="num">1</span><span class="label">Vector add prototype</span><span class="state state-done">Done</span></div>
        <div class="step"><span class="num">1.5</span><span class="label">Memory, scheduling & profiling</span><span class="state state-done">Done</span></div>
        <div class="step"><span class="num">2</span><span class="label">Tensor ops</span><span class="state state-done">Done</span></div>
        <div class="step"><span class="num">2.5</span><span class="label">Device-resident memory & pooled allocator</span><span class="state state-done">Done</span></div>
        <div class="step"><span class="num">3</span><span class="label">Vendor GEMM acceleration</span><span class="state state-done">Done</span></div>
        <div class="step"><span class="num">4</span><span class="label">SaC compiler integration</span><span class="state state-pending">Pending</span></div>
        <div class="step"><span class="num">5</span><span class="label">Training step</span><span class="state state-pending">Pending</span></div>
      </div>
      <div class="card roadmap-track">
        <h3>Phase track</h3><p class="track-sub">milestone.md §22 as amended by edge_inference_milestone_insert.md §8 — a separate development-phase breakdown, not the Milestone track's own "4"/"5"</p>
        <div class="step"><span class="num">0</span><span class="label">Repository & build skeleton</span><span class="state state-done">Done</span></div>
        <div class="step"><span class="num">1</span><span class="label">Runtime prototype</span><span class="state state-done">Done</span></div>
        <div class="step"><span class="num">2</span><span class="label">Common GPU/Tensor IR</span><span class="state state-done">Done</span></div>
        <div class="step"><span class="num">2.5</span><span class="label">Device-resident memory & pooled allocator</span><span class="state state-done">Done</span></div>
        <div class="step"><span class="num">3</span><span class="label">OpenCL backend</span><span class="state state-done">Done</span></div>
        <div class="step"><span class="num">3.5</span><span class="label">Vendor GEMM for existing backends</span><span class="state state-done">Done</span></div>
        <div class="step"><span class="num">4</span><span class="label">Local/edge inference runtime vocabulary</span><span class="state state-done">Done</span></div>
        <div class="step"><span class="num">4.5</span><span class="label">Vulkan backend prototype</span><span class="state state-pending">Pending</span></div>
        <div class="step"><span class="num">5</span><span class="label">Inference benchmark suite (hardware tiers)</span><span class="state state-pending">Pending</span></div>
        <div class="step"><span class="num">5.5</span><span class="label">llama.cpp/ggml/vLLM compatibility investigation</span><span class="state state-pending">Pending</span></div>
        <div class="step"><span class="num">6</span><span class="label">SYCL backend</span><span class="state state-pending">Pending</span></div>
        <div class="step"><span class="num">6.5</span><span class="label">SYCL vendor library integration</span><span class="state state-pending">Pending</span></div>
        <div class="step"><span class="num">7</span><span class="label">HIP backend</span><span class="state state-pending">Pending</span></div>
        <div class="step"><span class="num">7.5</span><span class="label">HIP vendor library integration</span><span class="state state-pending">Pending</span></div>
        <div class="step"><span class="num">8</span><span class="label">Backend auto-selector</span><span class="state state-partial">Partial</span></div>
        <div class="step"><span class="num">9</span><span class="label">Autodiff module</span><span class="state state-pending">Pending</span></div>
        <div class="step"><span class="num">10</span><span class="label">Mixed precision</span><span class="state state-pending">Pending</span></div>
      </div>
    </div>
  </section>

  <footer>
    <div class="verify-cmds">
      <div>ctest --test-dir build --output-on-failure</div>
      <div>ldd build/gpubridge_vecadd_test | grep -i opencl &nbsp;→&nbsp; (empty)</div>
      <div>readelf -d build/gpubridge_vecadd_test | grep NEEDED &nbsp;→&nbsp; libc.so.6 only</div>
    </div>
    <p class="colophon">Generated by scripts/bench_report.py from a live ctest + GPUBRIDGE_PROFILE=1 run. Run it again after changes to refresh this page and extend the history chart.</p>
  </footer>

</div>

<script>
  const RUN_HISTORY = {history_json};
  const GENERATED_AT = "{generated_at}";

  // Polls this same file (file:// or http://) via XHR and reloads the page
  // once bench_report.py has rewritten it with a newer run_id — so a tab
  // left open picks up a fresh `python3 scripts/bench_report.py` run without
  // a manual browser refresh. Silently gives up polling if the request
  // itself fails (e.g. a browser that blocks XHR to file://); this is a
  // convenience layer on top of the "always works as a static file" page,
  // not a requirement for it.
  (function watchForUpdates() {{
    const POLL_MS = 3000;
    function poll() {{
      let xhr;
      try {{
        xhr = new XMLHttpRequest();
        xhr.open("GET", location.pathname + "?_=" + Date.now(), true);
      }} catch (e) {{
        return; // can't construct/open a request here; stop polling
      }}
      xhr.onload = function () {{
        const body = xhr.responseText || "";
        const m = body.match(/const GENERATED_AT = "([^"]*)";/);
        if (m && m[1] && m[1] !== GENERATED_AT) {{
          location.reload();
          return;
        }}
        setTimeout(poll, POLL_MS);
      }};
      xhr.onerror = function () {{}}; // stop polling on the first failure
      xhr.send();
    }}
    setTimeout(poll, POLL_MS);
  }})();

  function renderHistoryChart() {{
    const el = document.getElementById("history-chart");
    if (RUN_HISTORY.length === 0) {{
      el.innerHTML = '<p class="history-empty">No runs recorded yet.</p>';
      return;
    }}
    const w = el.clientWidth || 600, h = 180, padL = 44, padR = 12, padT = 10, padB = 24;
    const values = [];
    RUN_HISTORY.forEach(p => {{
      if (typeof p.cpu_total_ms === "number") values.push(p.cpu_total_ms);
      if (typeof p.auto_total_ms === "number") values.push(p.auto_total_ms);
    }});
    const maxV = Math.max(...values, 0.01);
    const n = RUN_HISTORY.length;
    const x = i => n === 1 ? (padL + (w - padL - padR) / 2) : padL + (i / (n - 1)) * (w - padL - padR);
    const y = v => padT + (1 - (v / maxV)) * (h - padT - padB);

    const styles = getComputedStyle(document.documentElement);
    const accent = styles.getPropertyValue("--accent").trim();
    const accent2 = styles.getPropertyValue("--accent-2").trim();
    const line = styles.getPropertyValue("--line").trim();
    const muted = styles.getPropertyValue("--muted").trim();
    const surface = styles.getPropertyValue("--surface").trim();

    function seriesPath(key) {{
      const pts = RUN_HISTORY.map((p, i) => typeof p[key] === "number" ? [x(i), y(p[key])] : null).filter(Boolean);
      if (pts.length === 0) return "";
      return "M " + pts.map(p => p.join(" ")).join(" L ");
    }}
    function seriesDots(key, color, labelKey) {{
      return RUN_HISTORY.map((p, i) => {{
        if (typeof p[key] !== "number") return "";
        const label = `${{p.run_id}} \\u00b7 ${{p[key].toFixed(2)}}ms${{labelKey ? " (" + p[labelKey] + ")" : ""}}`;
        return `<circle cx="${{x(i)}}" cy="${{y(p[key])}}" r="4" fill="${{color}}" stroke="${{surface}}" stroke-width="2"><title>${{label}}</title></circle>`;
      }}).join("");
    }}

    const gridY0 = y(0), gridYmax = y(maxV);
    const svg = `
      <svg viewBox="0 0 ${{w}} ${{h}}" width="100%" height="${{h}}" role="img" aria-label="Total GPU path time per run">
        <line x1="${{padL}}" y1="${{gridY0}}" x2="${{w - padR}}" y2="${{gridY0}}" stroke="${{line}}" stroke-width="1"/>
        <line x1="${{padL}}" y1="${{gridYmax}}" x2="${{w - padR}}" y2="${{gridYmax}}" stroke="${{line}}" stroke-width="1"/>
        <text x="${{padL - 8}}" y="${{gridY0 + 4}}" text-anchor="end" font-size="10" fill="${{muted}}" font-family="var(--mono)">0</text>
        <text x="${{padL - 8}}" y="${{gridYmax + 4}}" text-anchor="end" font-size="10" fill="${{muted}}" font-family="var(--mono)">${{maxV.toFixed(1)}}</text>
        <path d="${{seriesPath('auto_total_ms')}}" fill="none" stroke="${{accent}}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
        <path d="${{seriesPath('cpu_total_ms')}}" fill="none" stroke="${{accent2}}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
        ${{seriesDots('auto_total_ms', accent, 'auto_backend')}}
        ${{seriesDots('cpu_total_ms', accent2)}}
      </svg>`;
    el.innerHTML = svg;
  }}
  renderHistoryChart();
  window.addEventListener("resize", renderHistoryChart);
</script>
"""


def main():
    print(f"Running ctest + profiled binaries in {BUILD_DIR} ...")
    record = collect_run_record()
    append_history(record)
    print(f"Recorded run {record['run_id']}: {record['ctest']['passed']}/{record['ctest']['total']} tests passing")

    history = load_history()
    html = render_dashboard(history)
    DASHBOARD_OUT.write_text(html)
    print(f"Wrote {DASHBOARD_OUT} ({len(history)} run{'s' if len(history) != 1 else ''} in history)")


if __name__ == "__main__":
    main()
