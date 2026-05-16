from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path


LOGGER = logging.getLogger(__name__)


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
    LOGGER.info(
        "TabReD preparation started repo=%s output_root=%s datasets=%s clone_if_missing=%s link=%s",
        repo,
        output,
        selected,
        clone_if_missing,
        link,
    )

    if clone_if_missing and not repo.exists():
        repo.parent.mkdir(parents=True, exist_ok=True)
        LOGGER.info("cloning TabReD repository url=%s target=%s", TABRED_REPO_URL, repo)
        _run_logged_command(["git", "clone", "--depth", "1", TABRED_REPO_URL, str(repo)], label="tabred-git-clone")
        LOGGER.info("TabReD repository cloned target=%s", repo)
    if not repo.exists():
        LOGGER.error("TabReD repository missing path=%s", repo)
        raise FileNotFoundError(f"TabReD repository not found: {repo}")

    LOGGER.info("checking TabReD preprocessing runtime repo=%s", repo)
    _check_tabred_runtime(repo)
    output.mkdir(parents=True, exist_ok=True)
    LOGGER.info("TabReD output root ready path=%s", output)

    results: list[TabReDPrepareResult] = []
    for dataset in selected:
        script = TABRED_DATASETS[dataset]
        source = repo / "data" / dataset
        target = output / dataset
        LOGGER.info(
            "TabReD dataset preparation started dataset=%s script=%s source=%s target=%s",
            dataset,
            script,
            source,
            target,
        )

        if not source.exists():
            try:
                LOGGER.info("running TabReD preprocessing script dataset=%s script=%s", dataset, script)
                _run_logged_command(
                    [sys.executable, f"preprocessing/{script}"],
                    cwd=repo,
                    env=_tabred_env(repo),
                    label=f"tabred-{dataset}",
                )
                LOGGER.info("TabReD preprocessing script completed dataset=%s", dataset)
            except subprocess.CalledProcessError as exc:
                LOGGER.exception("TabReD preprocessing script failed dataset=%s exit_code=%s", dataset, exc.returncode)
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
        else:
            LOGGER.info("TabReD processed source already exists dataset=%s source=%s", dataset, source)

        if not _looks_like_tabred_dataset(source):
            LOGGER.error("TabReD expected files missing dataset=%s source=%s", dataset, source)
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

        LOGGER.info("linking/copying TabReD dataset dataset=%s link=%s", dataset, link)
        _link_or_copy(source, target, link=link)
        LOGGER.info("TabReD dataset ready dataset=%s target=%s", dataset, target)
        results.append(
            TabReDPrepareResult(
                dataset=dataset,
                status="completed",
                source_path=str(source),
                target_path=str(target),
                message="ready",
            )
        )

    LOGGER.info("TabReD preparation completed datasets=%d", len(results))
    return results


def validate_tabred_root(root: str | Path = "data/raw/tabred") -> list[TabReDPrepareResult]:
    root = Path(root)
    LOGGER.info("validating TabReD root path=%s", root)
    results = []
    for dataset in TABRED_DATASETS:
        target = root / dataset
        status = "completed" if _looks_like_tabred_dataset(target) else "missing"
        LOGGER.info("TabReD validation dataset=%s status=%s target=%s", dataset, status, target)
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


def _run_logged_command(
    command: list[str],
    *,
    label: str,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> None:
    LOGGER.info("subprocess started label=%s command=%s cwd=%s", label, command, cwd or Path.cwd())
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    if process.stdout is not None:
        for line in process.stdout:
            text = line.rstrip()
            if text:
                LOGGER.info("subprocess[%s] %s", label, text)
    return_code = process.wait()
    if return_code:
        LOGGER.error("subprocess failed label=%s return_code=%d", label, return_code)
        raise subprocess.CalledProcessError(return_code, command)
    LOGGER.info("subprocess completed label=%s return_code=%d", label, return_code)


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
            LOGGER.warning("TabReD dependency missing module=%s", module)
        else:
            LOGGER.debug("TabReD dependency available module=%s", module)
    if missing:
        raise ImportError(
            "TabReD preprocessing dependencies are missing: "
            + ", ".join(missing)
            + ". Install them with `uv sync --extra tabred`."
        )
    if not _has_kaggle_credentials():
        LOGGER.error("Kaggle credentials missing for TabReD preprocessing")
        raise FileNotFoundError(
            "Kaggle credentials are missing. Put kaggle.json in ~/.kaggle/ or set "
            "KAGGLE_USERNAME and KAGGLE_KEY. You must also accept the relevant Kaggle "
            "competition/dataset rules before preprocessing can download TabReD."
        )
    if not (repo / "preprocessing").exists():
        LOGGER.error("TabReD preprocessing directory missing repo=%s", repo)
        raise FileNotFoundError(f"TabReD preprocessing directory is absent in {repo}")
    LOGGER.info("TabReD runtime check passed repo=%s", repo)


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
        LOGGER.info("TabReD target already exists target=%s", target)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    if link:
        try:
            target.symlink_to(source.resolve(), target_is_directory=True)
            LOGGER.info("created TabReD symlink source=%s target=%s", source, target)
            return
        except OSError as exc:
            LOGGER.warning("failed to create TabReD symlink source=%s target=%s error=%s", source, target, exc)
            pass
    shutil.copytree(source, target)
    LOGGER.info("copied TabReD dataset source=%s target=%s", source, target)
