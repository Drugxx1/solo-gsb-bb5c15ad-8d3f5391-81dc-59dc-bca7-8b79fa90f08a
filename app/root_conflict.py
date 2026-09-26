"""Diagnose minimal conflicting root-requirement sets.

When a full ``/resolve`` request is unsatisfiable, this module pinpoints a
*minimal* set of root requirements that is already unsatisfiable on its
own: the set itself cannot be resolved, yet removing any single entry
from it resolves.  Entries keep the original requirement text and the
input position, so duplicate requirements stay distinguishable.

Method
------
Deletion-based minimisation over real solver runs -- never by reading
back the conflict path of one backtracking branch:

1. Solve the full root set.  ``ok`` means the input is solvable and the
   conflicting set is empty; ``limit`` means the search budget was
   exhausted, so the outcome is *undetermined* rather than unsatisfiable.
2. Otherwise walk the root requirements once in input order; for each
   entry, re-solve with that entry temporarily removed.  If the reduced
   set is still unsatisfiable the entry is dropped for good; if it
   resolves, the entry is essential to the conflict and kept.

Requirement sets are monotone (removing a requirement can only relax the
problem), so the surviving set is guaranteed minimal: every single-entry
removal was itself observed to resolve, and the surviving set was last
observed to conflict.  Any solver run that exhausts ``max_attempts``
aborts the diagnosis as undetermined -- hitting the search ceiling is
never reported as a conflict.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from packaging.version import Version

from .resolver import PackageIndex, ParsedDependency, SolveResult, resolve


@dataclass(frozen=True)
class RootItem:
    """One root requirement, tied to its original input token."""

    position: int  # index into the root requirements as supplied
    requirement: str  # original PEP 508 string, or "name+specifier" for mapping form
    name: str  # canonical package name
    specifier: str
    marker: Optional[str]
    dependency: ParsedDependency


def bind_root_items(raw: Any, parsed: list[ParsedDependency]) -> list[RootItem]:
    """Pair every parsed root requirement with its raw input and position.

    ``parse_requirements`` yields exactly one ParsedDependency per input
    item, in input order (mapping insertion order or list order), so the
    two sequences align positionally; malformed items already raised
    during parsing, before this function runs.
    """
    if isinstance(raw, dict):
        tokens = []
        for package, spec in raw.items():
            if spec is None:
                spec = ""
            tokens.append(f"{package}{spec}" if spec else str(package))
    elif isinstance(raw, list):
        tokens = [str(item) for item in raw]
    else:
        # Only reachable for root=null, which parses to zero requirements.
        tokens = []
    return [
        RootItem(
            position=position,
            requirement=token,
            name=dep.name,
            specifier=dep.specifier_text,
            marker=dep.marker_text,
            dependency=dep,
        )
        for position, (token, dep) in enumerate(zip(tokens, parsed))
    ]


@dataclass
class Diagnosis:
    status: str  # solvable | conflict | undetermined
    attempts: int  # candidate evaluations summed over every solver run
    rounds: int  # solver runs performed (1 initial + 1 per deletion test)
    items: Optional[list[RootItem]]  # minimal conflicting set; None if undetermined
    detail: Optional[str]  # explanation when undetermined


def diagnose_root_conflict(
    index: PackageIndex,
    items: list[RootItem],
    locks: dict[str, Version],
    *,
    allow_prereleases: bool = False,
    marker_env: Optional[dict[str, str]] = None,
    max_attempts: int = 100_000,
) -> Diagnosis:
    attempts = 0
    rounds = 0

    def run(subset: list[RootItem]) -> SolveResult:
        nonlocal attempts, rounds
        result = resolve(
            index,
            [item.dependency for item in subset],
            locks,
            allow_prereleases=allow_prereleases,
            marker_env=marker_env,
            max_attempts=max_attempts,
        )
        attempts += result.attempts
        rounds += 1
        return result

    first = run(items)
    if first.status == "ok":
        return Diagnosis("solvable", attempts, rounds, [], None)
    if first.status == "limit":
        return Diagnosis(
            status="undetermined",
            attempts=attempts,
            rounds=rounds,
            items=None,
            detail=(
                f"initial resolution exceeded {max_attempts} candidate "
                "evaluations; the input may still be satisfiable"
            ),
        )

    # Deletion-based minimisation in input (position) order: an entry is
    # dropped only when the remaining set still conflicts, so every entry
    # that survives is essential to the conflict.
    current = list(items)
    for item in items:
        trial = [kept for kept in current if kept.position != item.position]
        result = run(trial)
        if result.status == "limit":
            return Diagnosis(
                status="undetermined",
                attempts=attempts,
                rounds=rounds,
                items=None,
                detail=(
                    f"re-solving without root requirement #{item.position} "
                    f"({item.requirement!r}) exceeded {max_attempts} candidate "
                    "evaluations; the conflicting set could not be determined"
                ),
            )
        if result.status == "conflict":
            current = trial
    return Diagnosis("conflict", attempts, rounds, current, None)
