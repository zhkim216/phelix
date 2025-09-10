import pytest

from atomworks.ml.transforms.base import Compose, Identity, Transform, TransformPipelineError


class Transform1(Transform):
    incompatible_previous_transforms = ["Transform2"]

    def check_input(self, data):
        pass

    def forward(self, data):
        return data


class Transform2(Transform):
    def check_input(self, data):
        pass

    def forward(self, data):
        return data


class Transform3(Transform):
    def check_input(self, data):
        pass

    def forward(self, data):
        return data


class Transform4(Transform):
    requires_previous_transforms = ["Transform1", "Transform2", "Transform3"]
    previous_transforms_order_matters = True

    def check_input(self, data):
        pass

    def forward(self, data):
        return data


# Test all 4 cases
data = {"data": "data"}


class ErrorTransform(Transform):
    def check_input(self, data):
        pass

    def forward(self, data):
        raise ValueError("This transform always raises an error")


def test_incompatible_previous_transforms():
    with pytest.raises(TransformPipelineError):
        transform = Compose([Transform2(), Transform1()], track_rng_state=False)
        transform(data)


def test_missing_previous_transform():
    with pytest.raises(TransformPipelineError):
        transform = Compose([Transform1(), Transform4()], track_rng_state=False)
        transform(data)


def test_wrong_order_previous_transforms():
    with pytest.raises(TransformPipelineError):
        transform = Compose([Transform3(), Transform1(), Transform2(), Transform4()], track_rng_state=False)
        transform(data)


def test_success():
    transform = Compose([Transform1(), Transform2(), Transform3(), Transform4()], track_rng_state=False)
    transform(data)


def test_compose_error_handling():
    transform = Compose([ErrorTransform(), Transform1(), Transform2()], track_rng_state=False)
    with pytest.raises(ValueError) as excinfo:
        transform(data)

    assert "This transform always raises an error" in str(excinfo.value)


def test_compose_stop_before():
    transform = Compose([Transform1(), Transform2(), Transform3()], track_rng_state=False)
    result = transform(data, _stop_before="Transform3")

    history = result.__transform_history__
    assert len(history) == 2
    assert history[0]["name"] == "Transform1"
    assert history[1]["name"] == "Transform2"


def test_transform_addition():
    t1 = Transform1()
    t2 = Transform2()
    t3 = Transform3()

    # Test Transform + Transform
    combined = t1 + t2
    assert isinstance(combined, Compose)
    assert len(combined.transforms) == 2
    assert isinstance(combined.transforms[0], Transform1)
    assert isinstance(combined.transforms[1], Transform2)

    # Test Transform + Compose
    combined = t1 + Compose([t2, t3])
    assert isinstance(combined, Compose)
    assert len(combined.transforms) == 3
    assert isinstance(combined.transforms[0], Transform1)
    assert isinstance(combined.transforms[1], Transform2)
    assert isinstance(combined.transforms[2], Transform3)

    # Test Compose + Transform
    combined = Compose([t1, t2]) + t3
    assert isinstance(combined, Compose)
    assert len(combined.transforms) == 3
    assert isinstance(combined.transforms[0], Transform1)
    assert isinstance(combined.transforms[1], Transform2)
    assert isinstance(combined.transforms[2], Transform3)

    # Test Compose + Compose
    combined = Compose([t1, t2]) + Compose([t3, Identity()])
    assert isinstance(combined, Compose)
    assert len(combined.transforms) == 4
    assert isinstance(combined.transforms[0], Transform1)
    assert isinstance(combined.transforms[1], Transform2)
    assert isinstance(combined.transforms[2], Transform3)
    assert isinstance(combined.transforms[3], Identity)


if __name__ == "__main__":
    # For interactive debugging
    pytest.main(["-v", "-x", __file__])
