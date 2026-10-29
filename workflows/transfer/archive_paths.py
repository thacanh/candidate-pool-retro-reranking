"""Resolve private imported archives after root-folder organization."""
from pathlib import Path


def imported_archive(root: Path, name: str) -> Path:
    """Prefer organized storage; retain support for newly downloaded root files."""
    if Path(name).name != name or '/' in name or '\\' in name:
        raise ValueError('Archive name must be a basename')
    if name.startswith('N3_S4_'):
        group = 'N3_S4'
    elif name.startswith('N3_S'):
        group = 'N3'
    elif name.startswith('N4.'):
        group = 'N4'
    else:
        raise ValueError(f'Unknown imported archive family: {name}')
    organized = Path(root) / 'archive/imports/dd_final_round' / group / name
    legacy = Path(root) / name
    if organized.exists() and legacy.exists():
        raise FileExistsError(f'Ambiguous archive copies: {organized} and {legacy}')
    if organized.is_file():
        return organized
    if legacy.is_file():
        return legacy
    raise FileNotFoundError(organized)
