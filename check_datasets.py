"""
Check dataset setup -- verifica che i dataset siano presenti.

Esegui con:
    python check_datasets.py
"""

from pathlib import Path

ROOT = Path(__file__).parent
DATA = ROOT / 'data'


def check_kaggle():
    path = DATA / 'posture-keypoints-detection'
    if not path.exists():
        return False, f"NOT FOUND: {path}"
    imgs = [f for f in path.rglob('*') if f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp', '.webp')]
    if not imgs:
        return False, f"EMPTY: {path} (no images found)"
    return True, f"OK: {len(imgs)} images in {path}"


def check_assembly101():
    path = DATA / 'assembly101'
    if not path.exists():
        return False, f"NOT FOUND: {path}"
    vids = [f for f in path.rglob('*') if f.suffix.lower() in ('.mp4', '.avi', '.mov', '.mkv', '.webm')]
    if not vids:
        return False, f"EMPTY: {path} (no videos found)"
    return True, f"OK: {len(vids)} videos in {path}"


def main():
    print("=" * 55)
    print("ErgoVision -- Dataset check")
    print("=" * 55)
    print(f"Project root: {ROOT}")
    print()

    for label, fn in [
        ("Posture Keypoints (Kaggle)", check_kaggle),
        ("Assembly101", check_assembly101),
    ]:
        ok, msg = fn()
        tag = "OK" if ok else "--"
        print(f"  [{tag}] {label}")
        print(f"         {msg}")
        print()

    print("Links:")
    print("  Kaggle:      https://www.kaggle.com/datasets/melsmm/posture-keypoints-detection")
    print("  Assembly101: https://assembly-101.github.io/")
    print("  Download:    https://github.com/Assembly101-2022/assembly101-download")
    print()


if __name__ == "__main__":
    main()
