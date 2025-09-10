import pytest

from atomworks.ml.transforms.base import AddData, Compose, ConditionalRoute, Identity, RandomRoute, RemoveKeys
from atomworks.ml.utils.rng import create_rng_state_from_seeds, rng_state

RANDOM_ROUTE_TEST_CASES = [
    {
        "seed": 43,
        "expected_transform_history": ["AddData", "RandomRoute", "AddData"],
        "expected_data": {"test": "value", "test3": "value3"},
    },
    {
        "seed": 1,
        "expected_transform_history": ["AddData", "RandomRoute", "RemoveKeys", "AddData"],
        "expected_data": {"test3": "value3"},
    },
    {
        "seed": 4,
        "expected_transform_history": ["AddData", "RandomRoute", "AddData", "AddData"],
        "expected_data": {"test": "value", "test2": "value2", "test3": "value3"},
    },
]


@pytest.mark.parametrize("test_case", RANDOM_ROUTE_TEST_CASES)
def test_route_probabilistically(test_case):
    with rng_state(create_rng_state_from_seeds(np_seed=test_case["seed"])):
        pipe = Compose(
            [
                AddData(data={"test": "value"}),
                RandomRoute(
                    transforms=[Identity(), RemoveKeys(keys=["test"]), AddData({"test2": "value2"})],
                    probs=[0.3, 0.5, 0.2],
                ),
                AddData({"test3": "value3"}),
            ]
        )

        data = pipe({})

    history = [t["name"] for t in data.__transform_history__]
    assert history == test_case["expected_transform_history"]
    assert data == test_case["expected_data"]


# Define test cases
CONDITIONAL_ROUTE_TEST_CASES_WITH_STRING = [
    {
        "condition_value": "train",
        "expected_data": {"mode": "train", "status": "training"},
        "input_data": {"mode": "train"},
    },
    {
        "condition_value": "inference",
        "expected_data": {"mode": "inference", "status": "inference"},
        "input_data": {"mode": "inference"},
    },
    {
        "condition_value": "unknown",
        "expected_data": {"mode": "unknown"},  # Expect Identity, no 'status' key
        "input_data": {"mode": "unknown"},
    },
]


@pytest.mark.parametrize("test_case", CONDITIONAL_ROUTE_TEST_CASES_WITH_STRING)
def test_conditional_route_with_string(test_case):
    # Define the condition function
    condition_func = lambda data: data.get("mode", "default")  # noqa

    # Create the ConditionalRoute instance
    route = ConditionalRoute(
        condition_func=condition_func,
        transform_map={
            "train": AddData(data={"status": "training"}),
            "inference": AddData(data={"status": "inference"}),
            # Defaults to Identity if no match
        },
    )

    # Run the transform
    result_data = route(test_case["input_data"])

    # Check the result
    assert result_data == test_case["expected_data"]


# Define boolean-based test cases
CONDITIONAL_ROUTE_TEST_CASES_WITH_BOOLEAN = [
    {
        "condition_value": True,
        "expected_data": {"flag": True, "status": "active"},
        "input_data": {"flag": True},
    },
    {
        "condition_value": False,
        "expected_data": {"flag": False, "status": "inactive"},
        "input_data": {"flag": False},
    },
    {
        "condition_value": None,
        "expected_data": {"flag": None},  # Expect Identity, no 'status' key
        "input_data": {"flag": None},
    },
]


@pytest.mark.parametrize("test_case", CONDITIONAL_ROUTE_TEST_CASES_WITH_BOOLEAN)
def test_conditional_route_with_boolean(test_case):
    # Define the condition function for booleans
    def condition_func(data):
        return data.get("flag", None)

    # Create the ConditionalRoute instance
    route = ConditionalRoute(
        condition_func=condition_func,
        transform_map={
            True: AddData(data={"status": "active"}),
            False: AddData(data={"status": "inactive"}),
            # Defaults to Identity if no match
        },
    )

    # Run the transform
    result_data = route(test_case["input_data"])

    # Check the result
    assert result_data == test_case["expected_data"]


if __name__ == "__main__":
    pytest.main(["-v", "-x", __file__])
