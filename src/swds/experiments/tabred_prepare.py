from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path


TABRED_REPO_URL = "https://github.com/yandex-research/tabred.git"

TABRED_DATASETS = {
    "homesite-insurance": "homesite.py",
    "ecom-offers": "ecom-offers.py",
    "homecredit-default": "homecredit.py",
    "sberbank-housing": "sberbank-housing.py",
    "cooking-time": "cooking-time.py",
    "delivery-eta": "delivery-eta.py",
    "maps-routing": "maps-routing.py",
    "weather": "weather.py",
}


@dataclass(frozen=True)
class TabReDPrepareResult:
    dataset: str
    status: str
    source_path: str
    target_path: str
    message: str


def prepare_tabred(
    *,
    repo_dir: str | Path = "data/raw/tabred_repo",
    output_root: str | Path = "data/raw/tabred",
    datasets: list[str] | tuple[str, ...] = ("all",),
    clone_if_missing: bool = True,
    link: bool = True,
) -> list[TabReDPrepareResult]:
    repo = Path(repo_dir)
    output = Path(output_root)
    selected = _normalize_datasets(datasets)

    if clone_if_missing and not repo.exists():
        repo.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", TABRED_REPO_URL, str(repo)], check=True)
    if not repo.exists():
        raise FileNotFoundError(f"TabReD repository not found: {repo}")

    _check_tabred_runtime(repo)
    output.mkdir(parents=True, exist_ok=True)

    results: list[TabReDPrepareResult] = []
    for dataset in selected:
        script = TABRED_DATASETS[dataset]
        source = repo / "data" / dataset
        target = output / dataset

        if not source.exists():
            try:
                subprocess.run(
                    ["python", f"preprocessing/{script}"],
                    cwd=repo,
                    check=True,
                    env=_tabred_env(repo),
                )
            except subprocess.CalledProcessError as exc:
                results.append(
                    TabReDPrepareResult(
                        dataset=dataset,
                        status="failed",
                        source_path=str(source),
                        target_path=str(target),
                        message=f"preprocessing script failed with exit code {exc.returncode}",
                    )
                )
                continue

        if not _looks_like_tabred_dataset(source):
            results.append(
                TabReDPrepareResult(
                    dataset=dataset,
                    status="failed",
                    source_path=str(source),
                    target_path=str(target),
                    message="preprocessing did not create expected TabReD files",
                )
            )
            continue

        _link_or_copy(source, target, link=link)
        results.append(
            TabReDPrepareResult(
                dataset=dataset,
                status="completed",
                source_path=str(source),
                target_path=str(target),
                message="ready",
            )
        )

    return results


def validate_tabred_root(root: str | Path = "data/raw/tabred") -> list[TabReDPrepareResult]:
    root = Path(root)
    results = []
    for dataset in TABRED_DATASETS:
        target = root / dataset
        status = "completed" if _looks_like_tabred_dataset(target) else "missing"
        results.append(
            TabReDPrepareResult(
                dataset=dataset,
                status=status,
                source_path="",
                target_path=str(target),
                message="ready" if status == "completed" else "expected TabReD files are absent",
            )
        )
    return results


def _normalize_datasets(datasets: list[str] | tuple[str, ...]) -> list[str]:
    if not datasets or "all" in datasets:
        return list(TABRED_DATASETS)
    unknown = sorted(set(datasets) - set(TABRED_DATASETS))
    if unknown:
        raise ValueError(f"unknown TabReD datasets: {unknown}")
    return list(datasets)


def _check_tabred_runtime(repo: Path) -> None:
    missing = []
    for module in ("kaggle", "polars", "loguru", "openpyxl"):
        if find_spec(module) is None:
            missing.append(module)
    if missing:
        raise ImportError(
            "TabReD preprocessing dependencies are missing: "
            + ", ".join(missing)
            + ". Install them with `uv sync --extra tabred`."
        )
    if not _has_kaggle_credentials():
        raise FileNotFoundError(
            "Kaggle credentials are missing. Put kaggle.json in ~/.kaggle/ or set "
            "KAGGLE_USERNAME and KAGGLE_KEY. You must also accept the relevant Kaggle "
            "competition/dataset rules before preprocessing can download TabReD."
        )
    if not (repo / "preprocessing").exists():
        raise FileNotFoundError(f"TabReD preprocessing directory is absent in {repo}")


def _has_kaggle_credentials() -> bool:
    return (
        Path.home().joinpath(".kaggle", "kaggle.json").exists()
        or (bool(os.environ.get("KAGGLE_USERNAME")) and bool(os.environ.get("KAGGLE_KEY")))
    )


def _tabred_env(repo: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo) + os.pathsep + env.get("PYTHONPATH", "")
    return env


def _looks_like_tabred_dataset(path: Path) -> bool:
    return (
        path.exists()
        and (path / "Y.npy").exists()
        and (path / "info.json").exists()
        and (path / "split-default" / "train_idx.npy").exists()
        and (path / "split-default" / "val_idx.npy").exists()
        and (path / "split-default" / "test_idx.npy").exists()
        and any((path / f"{name}.npy").exists() for name in ("X_num", "X_bin", "X_cat", "X_meta"))
    )


def _link_or_copy(source: Path, target: Path, *, link: bool) -> None:
    if target.exists() or target.is_symlink():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    if link:
        try:
            target.symlink_to(source.resolve(), target_is_directory=True)
            return
        except OSError:
            pass
    shutil.copytree(source, target)
