from scripts.export.room_accuracy import validate_rooms


def _room(room_id, width, length, height):
    return {
        "room_id": room_id,
        "width_mm": width,
        "length_mm": length,
        "height_mm": height,
    }


def test_every_room_and_every_dimension_must_be_within_ten_mm():
    controls = {"rooms": [
        _room("living", 4947, 6540, 2697),
        _room("bedroom_1", 3030, 4890, 2740),
    ]}
    measured = {"rooms": [
        _room("living", 4955, 6531, 2707),
        _room("bedroom_1", 3030, 4890, 2740),
    ]}
    report = validate_rooms(measured, controls, tolerance_mm=10)
    assert report["status"] == "pass"
    assert report["rooms_passed"] == 2


def test_width_and_length_allow_a_ninety_degree_registration():
    controls = {"rooms": [_room("living", 4947, 6540, 2697)]}
    measured = {"rooms": [_room("living", 6540, 4947, 2697)]}
    assert validate_rooms(measured, controls, 10)["status"] == "pass"


def test_one_room_one_millimetre_beyond_tolerance_blocks_release():
    controls = {"rooms": [_room("living", 4947, 6540, 2697)]}
    measured = {"rooms": [_room("living", 4958, 6540, 2697)]}
    report = validate_rooms(measured, controls, 10)
    assert report["status"] == "fail"
    assert report["rooms"][0]["dimensions"]["short_span_mm"]["delta_mm"] == 11


def test_missing_height_or_room_blocks_release():
    controls = {"rooms": [
        _room("living", 4947, 6540, 2697),
        _room("bedroom_1", 3030, 4890, 2740),
    ]}
    measured = {"rooms": [_room("living", 4947, 6540, None)]}
    report = validate_rooms(measured, controls, 10)
    assert report["status"] == "fail"
    assert {r["status"] for r in report["rooms"]} == {"fail", "missing"}


def test_uncontrolled_extra_room_blocks_release():
    controls = {"rooms": [_room("living", 4947, 6540, 2697)]}
    measured = {"rooms": [
        _room("living", 4947, 6540, 2697),
        _room("mystery", 1000, 1000, 2500),
    ]}
    report = validate_rooms(measured, controls, 10)
    assert report["status"] == "fail"
    assert report["uncontrolled_measured_room_ids"] == ["mystery"]
