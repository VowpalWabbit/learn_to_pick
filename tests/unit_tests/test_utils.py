from typing import Any, List


class MockEncoder:
    def encode(self, to_encode: str) -> str:
        return [float(len(to_encode)), 0.0]


class MockEncoderReturnsList:
    def encode(self, to_encode: Any) -> List:
        if isinstance(to_encode, str):
            return [1.0, 2.0]
        elif isinstance(to_encode, List):
            return [[1.0, 2.0] for _ in range(len(to_encode))]
        raise ValueError("Invalid input type for unit test")


def assert_vw_ex_equals(first, second):
    first = first.split("\n")
    second = second.split("\n")
    assert len(first) == len(second)
    for _first, _second in zip(first, second):
        assert _first.strip() == _second.strip()
