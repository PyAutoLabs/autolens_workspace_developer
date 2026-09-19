"""Local dataset lookups follow the workspace's checkout placement."""

from pathlib import Path

from _repo_path import repo_path


def test_galaxy_dataset_lookup_in_grouped_and_flat_contexts(tmp_path):
    (tmp_path / ".pyauto-root").touch()
    brain = tmp_path / "PyAutoBrain" / "agents"
    brain.mkdir(parents=True)
    (brain / "_repo_paths.py").write_text(
        "from pathlib import Path\n"
        "def repo_path(root, name, required=False):\n"
        "    flat = Path(root) / name\n"
        "    return flat if flat.is_dir() else next(p for p in Path(root).glob('*/' + name) if p.is_dir())\n"
    )
    developer = tmp_path / "lens" / "autolens_workspace_developer"
    developer.mkdir(parents=True)
    galaxy = tmp_path / "galaxy" / "autogalaxy_workspace"
    galaxy.mkdir(parents=True)
    (galaxy / ".git").touch()
    assert repo_path(developer, "autogalaxy_workspace") == galaxy
    galaxy.rename(tmp_path / "autogalaxy_workspace")
    assert repo_path(developer, "autogalaxy_workspace") == tmp_path / "autogalaxy_workspace"


def test_standalone_flat_checkout_needs_no_brain(tmp_path):
    developer = tmp_path / "autolens_workspace_developer"
    developer.mkdir()
    galaxy = tmp_path / "autogalaxy_workspace"
    galaxy.mkdir()
    (galaxy / ".git").touch()
    assert repo_path(developer, "autogalaxy_workspace") == galaxy
