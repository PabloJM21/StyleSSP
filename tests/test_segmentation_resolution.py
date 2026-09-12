from pathlib import Path

from scripts.batch_canny_depth_control import resolve_struct_seg_dict


def test_resolve_struct_seg_dict_returns_none_when_missing(tmp_path: Path) -> None:
    image_path = tmp_path / "content.jpg"
    image_path.write_bytes(b"fake jpeg")

    assert resolve_struct_seg_dict(image_path) is None


def test_resolve_struct_seg_dict_finds_matching_pth(tmp_path: Path) -> None:
    image_path = tmp_path / "content.jpg"
    image_path.write_bytes(b"fake jpeg")
    seg_path = tmp_path / "content.pth"
    seg_path.write_bytes(b"seg")

    assert resolve_struct_seg_dict(image_path) == seg_path
