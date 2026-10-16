"""M4 / F2: tuples.jsonl parsing and validation."""

import json
import os

import pytest
from PIL import Image

from extensions.slider_tuple.manifest import ManifestError, load_manifest


def img(folder, name, size=(64, 64)):
    path = os.path.join(folder, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.new("RGB", size).save(path)
    return name


def write(folder, *tuples, name="tuples.jsonl"):
    path = os.path.join(folder, name)
    with open(path, "w") as f:
        for t in tuples:
            f.write((t if isinstance(t, str) else json.dumps(t)) + "\n")
    return path


def basic(folder, tid="p01", scales=(-1.0, 0.0, 1.0)):
    return {
        "id": tid,
        "caption": "photo of a person",
        "members": [{"image": img(folder, f"{tid}/m{i}.png"), "scale": s} for i, s in enumerate(scales)],
    }


def test_parses_paths_scales_and_optional_fields(tmp_path):
    t = basic(tmp_path)
    t["members"][2]["weight"] = 0.5
    t["members"][2]["mask"] = img(tmp_path, "p01/mask.png")
    (parsed,) = load_manifest(write(tmp_path, t, "", basic(tmp_path, "p02")))[:1]
    assert parsed.id == "p01" and parsed.caption == "photo of a person"
    assert [m.scale for m in parsed.members] == [-1.0, 0.0, 1.0]
    assert parsed.members[0].image == os.path.join(tmp_path, "p01", "m0.png")
    assert parsed.members[2].weight == 0.5 and parsed.members[2].mask.endswith("mask.png")
    assert parsed.anchor is parsed.members[0]
    assert parsed.trainable == [0, 2]
    assert parsed.size == (64, 64)
    assert len(load_manifest(os.path.join(tmp_path, "tuples.jsonl"))) == 2  # blank line skipped


def test_unequal_member_sizes_rejected(tmp_path):
    t = basic(tmp_path)
    t["members"][1]["image"] = img(tmp_path, "p01/big.png", size=(64, 96))
    with pytest.raises(ManifestError, match="share one pixel size"):
        load_manifest(write(tmp_path, t))


def test_missing_scale_rejected(tmp_path):
    t = basic(tmp_path)
    del t["members"][0]["scale"]
    with pytest.raises(ManifestError, match="needs 'image' and 'scale'"):
        load_manifest(write(tmp_path, t))


@pytest.mark.parametrize("scales", [(0.0, 0.0), (0.0, 0.0, 1.0)])
def test_tuples_without_two_nonzero_members_rejected(tmp_path, scales):
    with pytest.raises(ManifestError, match="at least two members with a nonzero scale"):
        load_manifest(write(tmp_path, basic(tmp_path, scales=scales)))


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda t: t.update(extra=1), "unknown keys"),
        (lambda t: t["members"][0].update(scael=1), "unknown keys"),
        (lambda t: t.pop("caption"), "missing 'caption'"),
        (lambda t: t.update(caption="  "), "non-empty string"),
        (lambda t: t.update(members=t["members"][:1]), "at least two members"),
        (lambda t: t["members"][0].update(scale="1"), "expected a number"),
        (lambda t: t["members"][0].update(scale=True), "expected a number"),
        (lambda t: t["members"][0].update(weight=-1), "must be >= 0"),
        (lambda t: t["members"][0].update(image="nope.png"), "file not found"),
        (lambda t: t["members"][1].update(image=t["members"][0]["image"]), "listed twice"),
    ],
)
def test_invalid_tuples_rejected(tmp_path, mutate, message):
    t = basic(tmp_path)
    mutate(t)
    with pytest.raises(ManifestError, match=message):
        load_manifest(write(tmp_path, t))


def test_file_level_errors(tmp_path):
    with pytest.raises(ManifestError, match="not found"):
        load_manifest(os.path.join(tmp_path, "missing.jsonl"))
    with pytest.raises(ManifestError, match="no tuples"):
        load_manifest(write(tmp_path, "", name="empty.jsonl"))
    with pytest.raises(ManifestError, match=r":1: invalid JSON"):
        load_manifest(write(tmp_path, "{not json", name="bad.jsonl"))
    with pytest.raises(ManifestError, match="duplicate id"):
        load_manifest(write(tmp_path, basic(tmp_path), basic(tmp_path), name="dup.jsonl"))
    a, b = basic(tmp_path, "a"), basic(tmp_path, "b")
    b["members"][0]["image"] = a["members"][0]["image"]
    with pytest.raises(ManifestError, match="same first \\(anchor\\) image"):
        load_manifest(write(tmp_path, a, b, name="anchor.jsonl"))


def test_absolute_paths_accepted(tmp_path):
    t = basic(tmp_path)
    for m in t["members"]:
        m["image"] = os.path.join(tmp_path, m["image"])
    other = tmp_path / "elsewhere"
    other.mkdir()
    assert load_manifest(write(other, t))[0].members[0].image == t["members"][0]["image"]
