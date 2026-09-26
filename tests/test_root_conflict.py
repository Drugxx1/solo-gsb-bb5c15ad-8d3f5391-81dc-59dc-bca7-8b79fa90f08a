"""End-to-end tests for POST /diagnose/root-conflict."""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _diagnose(**payload):
    response = client.post("/diagnose/root-conflict", json=payload)
    return response.status_code, response.json()


def _resolve(**payload):
    response = client.post("/resolve", json=payload)
    return response.status_code, response.json()


CONFLICT_PACKAGES = {
    "a": {"1.0": {"dependencies": {"x": ">=2.0"}}},
    "b": {"1.0": {"dependencies": {"x": "<2.0"}}},
    "c": {"1.0": {"dependencies": {"y": ">=1.0"}}},
    "x": {"1.0": {}, "2.0": {}},
    "y": {"1.0": {}},
}


def _chain_packages():
    packages = {}
    for i in range(3):
        packages[f"p{i}"] = {
            f"{major}.0": {"dependencies": {f"p{i+1}": ""}}
            for major in range(1, 4)
        }
    packages["p3"] = {f"{major}.0": {} for major in range(1, 4)}
    return packages


def test_solvable_input_returns_empty_set():
    status, body = _diagnose(packages=CONFLICT_PACKAGES, root=["a", "c"])
    assert status == 200
    assert body["status"] == "solvable"
    assert body["conflicting_roots"] == []
    assert body["rounds"] == 1


def test_conflict_set_is_minimal_and_keeps_positions():
    status, body = _diagnose(packages=CONFLICT_PACKAGES, root=["a", "b", "c"])
    assert status == 200
    assert body["status"] == "conflict"
    # c is uninvolved in the x conflict and must be dropped.
    assert [e["position"] for e in body["conflicting_roots"]] == [0, 1]
    assert [e["requirement"] for e in body["conflicting_roots"]] == ["a", "b"]
    assert body["rounds"] == 4  # initial solve + one re-solve per root


def test_returned_set_conflicts_and_every_single_removal_resolves():
    status, body = _diagnose(
        packages=CONFLICT_PACKAGES,
        root=["a>=1.0", "b>=1.0", "c>=1.0"],
    )
    assert status == 200
    requirements = [e["requirement"] for e in body["conflicting_roots"]]
    assert len(requirements) == 2
    # The returned requirement strings are reusable as /resolve roots ...
    resolve_status, _ = _resolve(packages=CONFLICT_PACKAGES, root=requirements)
    assert resolve_status == 409  # ... the set itself is unsatisfiable ...
    for dropped in range(len(requirements)):
        reduced = requirements[:dropped] + requirements[dropped + 1:]
        resolve_status, _ = _resolve(packages=CONFLICT_PACKAGES, root=reduced)
        assert resolve_status == 200  # ... but every single removal resolves.


def test_duplicate_requirements_are_distinguished_by_position():
    status, body = _diagnose(
        packages={"x": {"1.0": {}, "2.0": {}}},
        root=["x>=2.0", "x>=2.0", "x<2.0"],
    )
    assert status == 200
    assert body["status"] == "conflict"
    entries = body["conflicting_roots"]
    # One of the two identical pins is redundant; positions tell them apart.
    assert [e["position"] for e in entries] == [1, 2]
    assert [e["requirement"] for e in entries] == ["x>=2.0", "x<2.0"]


def test_mapping_form_preserves_original_requirement():
    status, body = _diagnose(
        packages=CONFLICT_PACKAGES,
        root={"a": ">=1.0", "b": ""},
    )
    assert status == 200
    assert body["status"] == "conflict"
    assert body["conflicting_roots"] == [
        {
            "position": 0,
            "requirement": "a>=1.0",
            "name": "a",
            "specifier": ">=1.0",
            "marker": None,
        },
        {
            "position": 1,
            "requirement": "b",
            "name": "b",
            "specifier": "",
            "marker": None,
        },
    ]


def test_marker_and_original_string_are_preserved():
    root = ["a>=1.0; sys_platform == 'linux'", "b"]
    status, body = _diagnose(packages=CONFLICT_PACKAGES, root=root)
    assert status == 200
    assert body["status"] == "conflict"
    entry = body["conflicting_roots"][0]
    assert entry["requirement"] == "a>=1.0; sys_platform == 'linux'"
    assert entry["name"] == "a"
    assert entry["specifier"] == ">=1.0"
    assert entry["marker"] == 'sys_platform == "linux"'

    # With the marker evaluated false the requirement is inert: no conflict.
    status, body = _diagnose(
        packages=CONFLICT_PACKAGES,
        root=root,
        marker_env={"sys_platform": "win32"},
    )
    assert status == 200
    assert body["status"] == "solvable"
    assert body["conflicting_roots"] == []


def test_initial_search_limit_is_undetermined_not_conflict():
    status, body = _diagnose(
        packages=_chain_packages(),
        root=["p0"],
        max_attempts=3,
    )
    assert status == 200
    assert body["status"] == "undetermined"
    assert body["conflicting_roots"] is None
    assert body["rounds"] == 1
    assert "3" in body["detail"]


def test_search_limit_during_minimisation_is_undetermined():
    packages = _chain_packages()
    # "a" is absent from the index, so the full input conflicts well within
    # budget; re-solving without "a" walks the whole chain and hits the cap.
    status, body = _diagnose(packages=packages, root=["a", "p0"], max_attempts=3)
    assert status == 200
    assert body["status"] == "undetermined"
    assert body["conflicting_roots"] is None
    assert body["rounds"] == 2

    # With a generous budget the same input is a clean single-root conflict.
    status, body = _diagnose(packages=packages, root=["a", "p0"])
    assert status == 200
    assert body["status"] == "conflict"
    assert [e["position"] for e in body["conflicting_roots"]] == [0]
    assert body["conflicting_roots"][0]["requirement"] == "a"


def test_empty_root_is_solvable():
    status, body = _diagnose(packages={"x": {"1.0": {}}}, root={})
    assert status == 200
    assert body["status"] == "solvable"
    assert body["conflicting_roots"] == []


def test_locks_are_honoured_during_diagnosis():
    status, body = _diagnose(
        packages={"x": {"1.0": {}, "2.0": {}}},
        root={"x": ">=1.0"},
        locks={"x": "1.0"},
    )
    assert status == 200
    assert body["status"] == "solvable"


def test_invalid_input_uses_standard_error_handling():
    status, body = _diagnose(
        packages={"x": {"1.0": {"dependencies": {"y": ">>>1"}}}, "y": {"1.0": {}}},
        root={"x": ""},
    )
    assert status == 400
    assert body["error"]["code"] == "invalid_request"

    status, body = _diagnose(packages={"x": {"1.0": {}}}, root=["!!!"])
    assert status == 400
    assert body["error"]["code"] == "invalid_request"


def test_extra_fields_rejected():
    response = client.post(
        "/diagnose/root-conflict",
        json={"packages": {}, "root": {}, "unexpected": 1},
    )
    assert response.status_code == 422


def test_deterministic_results():
    payload = {"packages": CONFLICT_PACKAGES, "root": ["a", "b", "c"]}
    results = [
        client.post("/diagnose/root-conflict", json=payload).json()
        for _ in range(3)
    ]
    assert all(r == results[0] for r in results)
