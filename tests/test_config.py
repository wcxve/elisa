from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


def _parse_payload(stdout: str) -> dict[str, object]:
    lines = [line for line in stdout.splitlines() if line.strip()]
    if not lines:
        raise AssertionError('no stdout from subprocess')
    return json.loads(lines[-1])


def _run_subprocess(script: str) -> dict[str, object]:
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env.pop('XLA_FLAGS', None)
    env['JAX_PLATFORM_NAME'] = 'cpu'

    existing_pythonpath = env.get('PYTHONPATH')
    if existing_pythonpath:
        env['PYTHONPATH'] = f'{repo_root}{os.pathsep}{existing_pythonpath}'
    else:
        env['PYTHONPATH'] = str(repo_root)

    result = subprocess.run(
        [sys.executable, '-c', textwrap.dedent(script)],
        check=True,
        capture_output=True,
        text=True,
        cwd=repo_root,
        env=env,
    )
    return _parse_payload(result.stdout)


def _skip_if_needed(payload: dict[str, object]) -> None:
    if payload.get('skipped'):
        reason = payload.get('reason')
        if not isinstance(reason, str):
            reason = 'skipped by subprocess'
        pytest.skip(reason)


def _get_int(payload: dict[str, object], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str) and value.isdigit():
        return int(value)
    raise AssertionError(f'payload[{key!r}] is not an int')


def _get_str(payload: dict[str, object], key: str) -> str:
    value = payload.get(key)
    if isinstance(value, str):
        return value
    if value is None:
        return ''
    return str(value)


def test_default_import_device_count():
    """Ensure default import sets JAX device count on CPU."""
    script = """\
    import json
    import multiprocessing as mp
    import os
    import sys

    os.environ.pop("XLA_FLAGS", None)
    os.environ["JAX_PLATFORM_NAME"] = "cpu"

    import elisa
    import jax

    total_cores = mp.cpu_count()
    if total_cores < 2:
        print(
            json.dumps(
                {
                    "skipped": True,
                    "reason": "cpu_count < 2",
                    "total_cores": total_cores,
                }
            )
        )
        sys.exit(0)

    expected = 4 if total_cores >= 4 else total_cores - 1
    payload = {
        "count": jax.local_device_count(),
        "expected": expected,
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "total_cores": total_cores,
    }
    print(json.dumps(payload))
    """
    payload = _run_subprocess(script)
    _skip_if_needed(payload)

    count = _get_int(payload, 'count')
    expected = _get_int(payload, 'expected')
    xla_flags = _get_str(payload, 'xla_flags')
    assert count == expected
    assert f'--xla_force_host_platform_device_count={expected}' in xla_flags


def test_set_cpu_cores_after_default_import():
    """Ensure set_cpu_cores works after default import."""
    script = """\
    import json
    import multiprocessing as mp
    import os
    import sys

    os.environ.pop("XLA_FLAGS", None)
    os.environ["JAX_PLATFORM_NAME"] = "cpu"

    import elisa
    from elisa.util.config import set_cpu_cores

    total_cores = mp.cpu_count()
    if total_cores < 2:
        print(
            json.dumps(
                {
                    "skipped": True,
                    "reason": "cpu_count < 2",
                    "total_cores": total_cores,
                }
            )
        )
        sys.exit(0)

    expected = 2
    set_cpu_cores(expected)

    import jax

    payload = {
        "count": jax.local_device_count(),
        "expected": expected,
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "total_cores": total_cores,
    }
    print(json.dumps(payload))
    """
    payload = _run_subprocess(script)
    _skip_if_needed(payload)

    count = _get_int(payload, 'count')
    expected = _get_int(payload, 'expected')
    xla_flags = _get_str(payload, 'xla_flags')
    assert count == expected
    assert f'--xla_force_host_platform_device_count={expected}' in xla_flags
