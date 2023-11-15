from typing import List, Union

import pytest
from test_utils import MockEncoder

import learn_to_pick.base as base


def test_simple_context_str_no_emb() -> None:
    expected = {"a_namespace": {"raw": "test"}}

    featurized = base.embed("test", MockEncoder(), "a_namespace")
    assert featurized.sparse == expected
    assert featurized.dense == {}


def test_simple_context_str_w_emb() -> None:
    str1 = "test"
    expected_dense = {"a_namespace": [4.0, 0.0]}
    expected_sparse = {"a_namespace": {"raw": str1}}

    featurized = base.embed(base.Embed(str1), MockEncoder(), "a_namespace")
    assert featurized.dense == expected_dense
    assert featurized.sparse == {}

    featurized = base.embed(base.EmbedAndKeep(str1), MockEncoder(), "a_namespace")
    assert featurized.sparse == expected_sparse
    assert featurized.dense == expected_dense


def test_simple_context_str_w_nested_emb() -> None:
    # nested embeddings, innermost wins
    str1 = "test"
    expected_dense = {"a_namespace": [4.0, 0.0]}
    expected_sparse = {"a_namespace": {"raw": str1}}

    featurized = base.embed(
        base.EmbedAndKeep(base.Embed(str1)), MockEncoder(), "a_namespace"
    )
    assert featurized.dense == expected_dense
    assert featurized.sparse == {}

    featurized = base.embed(
        base.Embed(base.EmbedAndKeep(str1)), MockEncoder(), "a_namespace"
    )
    assert featurized.sparse == expected_sparse
    assert featurized.dense == expected_dense


def test_context_w_namespace_no_emb() -> None:
    expected_sparse = {"test_namespace": {"raw": "test"}}
    featurized = base.embed({"test_namespace": "test"}, MockEncoder())
    assert featurized.sparse == expected_sparse
    assert featurized.dense == {}


def test_context_w_namespace_w_emb() -> None:
    str1 = "test"
    expected_sparse = {"test_namespace": {"raw": str1}}
    expected_dense = {"test_namespace": [4.0, 0.0]}

    featurized = base.embed({"test_namespace": base.Embed(str1)}, MockEncoder())
    assert featurized.sparse == {}
    assert featurized.dense == expected_dense

    featurized = base.embed({"test_namespace": base.EmbedAndKeep(str1)}, MockEncoder())
    assert featurized.sparse == expected_sparse
    assert featurized.dense == expected_dense


def test_context_w_namespace_w_emb2() -> None:
    str1 = "test"
    expected_sparse = {"test_namespace": {"raw": str1}}
    expected_dense = {"test_namespace": [4.0, 0.0]}

    featurized = base.embed(base.Embed({"test_namespace": str1}), MockEncoder())
    assert featurized.sparse == {}
    assert featurized.dense == expected_dense

    featurized = base.embed(base.EmbedAndKeep({"test_namespace": str1}), MockEncoder())
    assert featurized.sparse == expected_sparse
    assert featurized.dense == expected_dense


def test_context_w_namespace_w_some_emb() -> None:
    str1 = "test"
    str2 = "test_"
    expected_sparse = {"test_namespace": {"raw": str1}}
    expected_dense = {"test_namespace2": [5.0, 0.0]}
    featurized = base.embed(
        {"test_namespace": str1, "test_namespace2": base.Embed(str2)}, MockEncoder()
    )
    assert featurized.sparse == expected_sparse
    assert featurized.dense == expected_dense

    expected_sparse = {
        "test_namespace": {"raw": str1},
        "test_namespace2": {"raw": str2},
    }
    featurized = base.embed(
        {"test_namespace": str1, "test_namespace2": base.EmbedAndKeep(str2)},
        MockEncoder(),
    )
    assert featurized.sparse == expected_sparse
    assert featurized.dense == expected_dense


def test_simple_action_strlist_no_emb() -> None:
    str1 = "test1"
    str2 = "test2"
    str3 = "test3"
    expected_sparse = [
        {"a_namespace": {"raw": str1}},
        {"a_namespace": {"raw": str2}},
        {"a_namespace": {"raw": str3}},
    ]
    to_embed: List[Union[str, base._Embed]] = [str1, str2, str3]
    featurized = base.embed(to_embed, MockEncoder(), "a_namespace")

    for i in range(len(featurized)):
        assert featurized[i].sparse == expected_sparse[i]
        assert featurized[i].dense == {}


def test_simple_action_strlist_w_emb() -> None:
    str1 = "test"
    str2 = "test_"
    str3 = "test__"

    expected_sparse = [
        {"a_namespace": {"raw": str1}},
        {"a_namespace": {"raw": str2}},
        {"a_namespace": {"raw": str3}},
    ]
    expected_dense = [
        {"a_namespace": [4.0, 0.0]},
        {"a_namespace": [5.0, 0.0]},
        {"a_namespace": [6.0, 0.0]},
    ]

    featurized = base.embed(
        base.Embed([str1, str2, str3]), MockEncoder(), "a_namespace"
    )
    for i in range(len(featurized)):
        assert featurized[i].sparse == {}
        assert featurized[i].dense == expected_dense[i]

    featurized = base.embed(
        base.EmbedAndKeep([str1, str2, str3]), MockEncoder(), "a_namespace"
    )
    for i in range(len(featurized)):
        assert featurized[i].sparse == expected_sparse[i]
        assert featurized[i].dense == expected_dense[i]


def test_simple_action_strlist_w_some_emb() -> None:
    str1 = "test"
    str2 = "test_"
    str3 = "test__"

    expected_sparse = [{"a_namespace": {"raw": str1}}, {}, {}]
    expected_dense = [{}, {"a_namespace": [5.0, 0.0]}, {"a_namespace": [6.0, 0.0]}]
    featurized = base.embed(
        [str1, base.Embed(str2), base.Embed(str3)], MockEncoder(), "a_namespace"
    )
    for i in range(len(featurized)):
        assert featurized[i].sparse == expected_sparse[i]
        assert featurized[i].dense == expected_dense[i]

    featurized = base.embed(
        [str1, base.EmbedAndKeep(str2), base.EmbedAndKeep(str3)],
        MockEncoder(),
        "a_namespace",
    )
    expected_sparse = [
        {"a_namespace": {"raw": str1}},
        {"a_namespace": {"raw": str2}},
        {"a_namespace": {"raw": str3}},
    ]
    for i in range(len(featurized)):
        assert featurized[i].sparse == expected_sparse[i]
        assert featurized[i].dense == expected_dense[i]


def test_action_w_namespace_no_emb() -> None:
    str1 = "test1"
    str2 = "test2"
    str3 = "test3"
    expected_sparse = [
        {"test_namespace": {"raw": str1}},
        {"test_namespace": {"raw": str2}},
        {"test_namespace": {"raw": str3}},
    ]

    featurized = base.embed(
        [
            {"test_namespace": str1},
            {"test_namespace": str2},
            {"test_namespace": str3},
        ],
        MockEncoder(),
    )
    for i in range(len(featurized)):
        assert featurized[i].sparse == expected_sparse[i]
        assert featurized[i].dense == {}


def test_action_w_namespace_w_emb() -> None:
    str1 = "test"
    str2 = "test_"
    str3 = "test__"
    expected_sparse = [
        {"test_namespace": {"raw": str1}},
        {"test_namespace": {"raw": str2}},
        {"test_namespace": {"raw": str3}},
    ]
    expected_dense = [
        {"test_namespace": [4.0, 0.0]},
        {"test_namespace": [5.0, 0.0]},
        {"test_namespace": [6.0, 0.0]},
    ]

    featurized = base.embed(
        [
            {"test_namespace": base.Embed(str1)},
            {"test_namespace": base.Embed(str2)},
            {"test_namespace": base.Embed(str3)},
        ],
        MockEncoder(),
    )
    for i in range(len(featurized)):
        assert featurized[i].sparse == {}
        assert featurized[i].dense == expected_dense[i]

    featurized = base.embed(
        [
            {"test_namespace": base.EmbedAndKeep(str1)},
            {"test_namespace": base.EmbedAndKeep(str2)},
            {"test_namespace": base.EmbedAndKeep(str3)},
        ],
        MockEncoder(),
    )
    for i in range(len(featurized)):
        assert featurized[i].sparse == expected_sparse[i]
        assert featurized[i].dense == expected_dense[i]


def test_action_w_namespace_w_emb2() -> None:
    str1 = "test"
    str2 = "test_"
    str3 = "test__"
    expected_sparse = [
        {"test_namespace1": {"raw": str1}},
        {"test_namespace2": {"raw": str2}},
        {"test_namespace3": {"raw": str3}},
    ]
    expected_dense = [
        {"test_namespace1": [4.0, 0.0]},
        {"test_namespace2": [5.0, 0.0]},
        {"test_namespace3": [6.0, 0.0]},
    ]

    featurized = base.embed(
        base.Embed(
            [
                {"test_namespace1": str1},
                {"test_namespace2": str2},
                {"test_namespace3": str3},
            ]
        ),
        MockEncoder(),
    )
    for i in range(len(featurized)):
        assert featurized[i].sparse == {}
        assert featurized[i].dense == expected_dense[i]

    featurized = base.embed(
        base.EmbedAndKeep(
            [
                {"test_namespace1": str1},
                {"test_namespace2": str2},
                {"test_namespace3": str3},
            ]
        ),
        MockEncoder(),
    )
    for i in range(len(featurized)):
        assert featurized[i].sparse == expected_sparse[i]
        assert featurized[i].dense == expected_dense[i]


def test_action_w_namespace_w_some_emb() -> None:
    str1 = "test"
    str2 = "test_"
    str3 = "test__"
    expected_sparse = [
        {"test_namespace": {"raw": str1}},
        {},
        {},
    ]
    expected_dense = [
        {},
        {"test_namespace": [5.0, 0.0]},
        {"test_namespace": [6.0, 0.0]},
    ]

    featurized = base.embed(
        [
            {"test_namespace": str1},
            {"test_namespace": base.Embed(str2)},
            {"test_namespace": base.Embed(str3)},
        ],
        MockEncoder(),
    )
    for i in range(len(featurized)):
        assert featurized[i].sparse == expected_sparse[i]
        assert featurized[i].dense == expected_dense[i]

    expected_sparse = [
        {"test_namespace": {"raw": str1}},
        {"test_namespace": {"raw": str2}},
        {"test_namespace": {"raw": str3}},
    ]
    featurized = base.embed(
        [
            {"test_namespace": str1},
            {"test_namespace": base.EmbedAndKeep(str2)},
            {"test_namespace": base.EmbedAndKeep(str3)},
        ],
        MockEncoder(),
    )
    for i in range(len(featurized)):
        assert featurized[i].sparse == expected_sparse[i]
        assert featurized[i].dense == expected_dense[i]


def test_action_w_namespace_w_emb_w_more_than_one_item_in_first_dict() -> None:
    str1 = "test"
    str2 = "test_"
    str3 = "test__"
    expected_sparse = [
        {"test_namespace2": {"raw": str1}},
        {"test_namespace2": {"raw": str2}},
        {"test_namespace2": {"raw": str3}},
    ]
    expected_dense = [
        {"test_namespace": [4.0, 0.0]},
        {"test_namespace": [5.0, 0.0]},
        {"test_namespace": [6.0, 0.0]},
    ]

    featurized = base.embed(
        [
            {"test_namespace": base.Embed(str1), "test_namespace2": str1},
            {"test_namespace": base.Embed(str2), "test_namespace2": str2},
            {"test_namespace": base.Embed(str3), "test_namespace2": str3},
        ],
        MockEncoder(),
    )
    for i in range(len(featurized)):
        assert featurized[i].sparse == expected_sparse[i]
        assert featurized[i].dense == expected_dense[i]

    expected_sparse = [
        {"test_namespace": {"raw": str1}, "test_namespace2": {"raw": str1}},
        {"test_namespace": {"raw": str2}, "test_namespace2": {"raw": str2}},
        {"test_namespace": {"raw": str3}, "test_namespace2": {"raw": str3}},
    ]
    featurized = base.embed(
        [
            {"test_namespace": base.EmbedAndKeep(str1), "test_namespace2": str1},
            {"test_namespace": base.EmbedAndKeep(str2), "test_namespace2": str2},
            {"test_namespace": base.EmbedAndKeep(str3), "test_namespace2": str3},
        ],
        MockEncoder(),
    )
    for i in range(len(featurized)):
        assert featurized[i].sparse == expected_sparse[i]
        assert featurized[i].dense == expected_dense[i]


def test_one_namespace_w_list_of_features_no_emb() -> None:
    str1 = "test1"
    str2 = "test2"
    expected_sparse = {
        "test_namespace_0": {"raw": str1},
        "test_namespace_1": {"raw": str2},
    }

    featurized = base.embed({"test_namespace": [str1, str2]}, MockEncoder())
    assert featurized.sparse == expected_sparse
    assert featurized.dense == {}


def test_one_namespace_w_list_of_features_w_some_emb() -> None:
    str1 = "test"
    str2 = "test_"
    expected_sparse = {"test_namespace_0": {"raw": str1}}
    expected_dense = {"test_namespace_1": [5.0, 0.0]}

    featurized = base.embed({"test_namespace": [str1, base.Embed(str2)]}, MockEncoder())
    assert featurized.sparse == expected_sparse
    assert featurized.dense == expected_dense


def test_nested_list_features_throws() -> None:
    with pytest.raises(ValueError):
        base.embed({"test_namespace": [[1, 2], [3, 4]]}, MockEncoder())


def test_dict_in_list_throws() -> None:
    with pytest.raises(ValueError):
        base.embed({"test_namespace": [{"a": 1}, {"b": 2}]}, MockEncoder())


def test_nested_dict_throws() -> None:
    with pytest.raises(ValueError):
        base.embed({"test_namespace": {"a": {"b": 1}}}, MockEncoder())


def test_list_of_tuples_throws() -> None:
    with pytest.raises(ValueError):
        base.embed({"test_namespace": [("a", 1), ("b", 2)]}, MockEncoder())
