import pytest
from test_utils import MockEncoder, assert_vw_ex_equals

import learn_to_pick.base as rl_chain
import learn_to_pick.pick_best as pick_best_chain
from learn_to_pick.pick_best import vw_cb_formatter


def test_pickbest_textembedder_missing_context_not_throws() -> None:
    featurizer = pick_best_chain.PickBestFeaturizer(
        auto_embed=False, model=MockEncoder()
    )
    named_action = {"action": ["0", "1", "2"]}
    event = pick_best_chain.PickBestEvent(
        inputs={}, to_select_from=named_action, based_on={}
    )
    featurizer.featurize(event)


def test_pickbest_textembedder_missing_actions_throws() -> None:
    featurizer = pick_best_chain.PickBestFeaturizer(
        auto_embed=False, model=MockEncoder()
    )
    event = pick_best_chain.PickBestEvent(
        inputs={}, to_select_from={}, based_on={"context": "context"}
    )
    with pytest.raises(ValueError):
        featurizer.featurize(event)


def test_pickbest_textembedder_no_label_no_emb() -> None:
    featurizer = pick_best_chain.PickBestFeaturizer(
        auto_embed=False, model=MockEncoder()
    )
    named_actions = {"action": ["0", "1", "2"]}
    expected = "\n".join(
        [
            "shared |context_sparse raw:=context",
            "|action_sparse raw:=0",
            "|action_sparse raw:=1",
            "|action_sparse raw:=2",
        ]
    )

    event = pick_best_chain.PickBestEvent(
        inputs={}, to_select_from=named_actions, based_on={"context": "context"}
    )
    vw_ex_str = vw_cb_formatter(*featurizer.featurize(event))
    assert_vw_ex_equals(vw_ex_str, expected)


def test_pickbest_textembedder_w_label_no_score_no_emb() -> None:
    featurizer = pick_best_chain.PickBestFeaturizer(
        auto_embed=False, model=MockEncoder()
    )
    named_actions = {"action": ["0", "1", "2"]}
    expected = "\n".join(
        [
            "shared |context_sparse raw:=context",
            "|action_sparse raw:=0",
            "|action_sparse raw:=1",
            "|action_sparse raw:=2",
        ]
    )
    selected = pick_best_chain.PickBestSelected(index=0, probability=1.0)
    event = pick_best_chain.PickBestEvent(
        inputs={},
        to_select_from=named_actions,
        based_on={"context": "context"},
        selected=selected,
    )
    vw_ex_str = vw_cb_formatter(*featurizer.featurize(event))
    assert_vw_ex_equals(vw_ex_str, expected)


def test_pickbest_textembedder_w_full_label_no_emb() -> None:
    featurizer = pick_best_chain.PickBestFeaturizer(
        auto_embed=False, model=MockEncoder()
    )
    named_actions = {"action": ["0", "1", "2"]}
    expected = "\n".join(
        [
            "shared |context_sparse raw:=context",
            "0:-0.0:1.0 |action_sparse raw:=0",
            "|action_sparse raw:=1",
            "|action_sparse raw:=2",
        ]
    )

    selected = pick_best_chain.PickBestSelected(index=0, probability=1.0, score=0.0)
    event = pick_best_chain.PickBestEvent(
        inputs={},
        to_select_from=named_actions,
        based_on={"context": "context"},
        selected=selected,
    )
    vw_ex_str = vw_cb_formatter(*featurizer.featurize(event))
    assert_vw_ex_equals(vw_ex_str, expected)


def test_pickbest_textembedder_w_full_label_w_emb() -> None:
    featurizer = pick_best_chain.PickBestFeaturizer(
        auto_embed=False, model=MockEncoder()
    )
    str1 = "0"
    str2 = "1"
    str3 = "2"

    ctx_str = "ctx"
    encoded_ctx_str = "0:3.0 1:0.0"

    named_actions = {"action": rl_chain.Embed([str1, str2, str3])}
    context = {"context": rl_chain.Embed(ctx_str)}
    expected = "\n".join(
        [
            f"shared |context_dense {encoded_ctx_str}",
            "0:-0.0:1.0 |action_dense 0:1.0 1:0.0",
            "|action_dense 0:1.0 1:0.0",
            "|action_dense 0:1.0 1:0.0",
        ]
    )  # noqa: E501
    selected = pick_best_chain.PickBestSelected(index=0, probability=1.0, score=0.0)
    event = pick_best_chain.PickBestEvent(
        inputs={}, to_select_from=named_actions, based_on=context, selected=selected
    )
    vw_ex_str = vw_cb_formatter(*featurizer.featurize(event))
    assert_vw_ex_equals(vw_ex_str, expected)


def test_pickbest_textembedder_w_full_label_w_embed_and_keep() -> None:
    featurizer = pick_best_chain.PickBestFeaturizer(
        auto_embed=False, model=MockEncoder()
    )
    str1 = "0"
    str2 = "1"
    str3 = "2"

    ctx_str = "ctx"
    encoded_ctx_str = "0:3.0 1:0.0"

    named_actions = {"action": rl_chain.EmbedAndKeep([str1, str2, str3])}
    context = {"context": rl_chain.EmbedAndKeep(ctx_str)}
    expected = "\n".join(
        [
            f"shared |context_dense {encoded_ctx_str} |context_sparse raw:={ctx_str}",
            "0:-0.0:1.0 |action_dense 0:1.0 1:0.0 |action_sparse raw:=0",
            "|action_dense 0:1.0 1:0.0 |action_sparse raw:=1",
            "|action_dense 0:1.0 1:0.0 |action_sparse raw:=2",
        ]
    )  # noqa: E501
    selected = pick_best_chain.PickBestSelected(index=0, probability=1.0, score=0.0)
    event = pick_best_chain.PickBestEvent(
        inputs={}, to_select_from=named_actions, based_on=context, selected=selected
    )
    vw_ex_str = vw_cb_formatter(*featurizer.featurize(event))
    assert_vw_ex_equals(vw_ex_str, expected)


def test_pickbest_textembedder_more_namespaces_no_label_no_emb() -> None:
    featurizer = pick_best_chain.PickBestFeaturizer(
        auto_embed=False, model=MockEncoder()
    )
    named_actions = {"action1": [{"a": "0", "b": "0"}, "1", "2"]}
    context = {"context1": "context1", "context2": "context2"}
    expected = "\n".join(
        [
            "shared |context1_sparse raw:=context1 |context2_sparse raw:=context2 ",
            "|a_sparse raw:=0 |b_sparse raw:=0",
            "|action1_sparse raw:=1",
            "|action1_sparse raw:=2",
        ]
    )  # noqa: E501
    event = pick_best_chain.PickBestEvent(
        inputs={}, to_select_from=named_actions, based_on=context
    )
    vw_ex_str = vw_cb_formatter(*featurizer.featurize(event))
    assert_vw_ex_equals(vw_ex_str, expected)


def test_pickbest_textembedder_more_namespaces_w_label_no_emb() -> None:
    featurizer = pick_best_chain.PickBestFeaturizer(
        auto_embed=False, model=MockEncoder()
    )
    named_actions = {"action": [{"a": "0", "b": "0"}, "1", "2"]}
    context = {"context1": "context1", "context2": "context2"}
    expected = "\n".join(
        [
            "shared |context1_sparse raw:=context1 |context2_sparse raw:=context2",
            "|a_sparse raw:=0 |b_sparse raw:=0",
            "|action_sparse raw:=1",
            "|action_sparse raw:=2",
        ]
    )  # noqa: E501
    selected = pick_best_chain.PickBestSelected(index=0, probability=1.0)
    event = pick_best_chain.PickBestEvent(
        inputs={}, to_select_from=named_actions, based_on=context, selected=selected
    )
    vw_ex_str = vw_cb_formatter(*featurizer.featurize(event))
    assert_vw_ex_equals(vw_ex_str, expected)


def test_pickbest_textembedder_more_namespaces_w_full_label_no_emb() -> None:
    featurizer = pick_best_chain.PickBestFeaturizer(
        auto_embed=False, model=MockEncoder()
    )
    named_actions = {"action": [{"a": "0", "b": "0"}, "1", "2"]}
    context = {"context1": "context1", "context2": "context2"}
    expected = "\n".join(
        [
            "shared |context1_sparse raw:=context1 |context2_sparse raw:=context2",
            "0:-0.0:1.0 |a_sparse raw:=0 |b_sparse raw:=0",
            "|action_sparse raw:=1",
            "|action_sparse raw:=2",
        ]
    )  # noqa: E501
    selected = pick_best_chain.PickBestSelected(index=0, probability=1.0, score=0.0)
    event = pick_best_chain.PickBestEvent(
        inputs={}, to_select_from=named_actions, based_on=context, selected=selected
    )
    vw_ex_str = vw_cb_formatter(*featurizer.featurize(event))
    assert_vw_ex_equals(vw_ex_str, expected)


def test_pickbest_textembedder_more_namespaces_w_full_label_w_full_emb() -> None:
    featurizer = pick_best_chain.PickBestFeaturizer(
        auto_embed=False, model=MockEncoder()
    )

    str1 = "0"
    str2 = "1"
    str3 = "2"

    ctx_str_1 = "ctx"
    ctx_str_2 = "ctx_"
    encoded_ctx_str_1 = "0:3.0 1:0.0"
    encoded_ctx_str_2 = "0:4.0 1:0.0"

    named_actions = {"action": rl_chain.Embed([{"a": str1, "b": str1}, str2, str3])}
    context = {
        "context1": rl_chain.Embed(ctx_str_1),
        "context2": rl_chain.Embed(ctx_str_2),
    }
    expected = "\n".join(
        [
            f"shared |context1_dense {encoded_ctx_str_1} |context2_dense {encoded_ctx_str_2}",
            f"0:-0.0:1.0 |a_dense 0:1.0 1:0.0 |b_dense 0:1.0 1:0.0",
            f"|action_dense 0:1.0 1:0.0",
            f"|action_dense 0:1.0 1:0.0",
        ]
    )  # noqa: E501

    selected = pick_best_chain.PickBestSelected(index=0, probability=1.0, score=0.0)
    event = pick_best_chain.PickBestEvent(
        inputs={}, to_select_from=named_actions, based_on=context, selected=selected
    )
    vw_ex_str = vw_cb_formatter(*featurizer.featurize(event))
    assert_vw_ex_equals(vw_ex_str, expected)


def test_pickbest_textembedder_more_namespaces_w_full_label_w_full_embed_and_keep() -> (
    None
):
    featurizer = pick_best_chain.PickBestFeaturizer(
        auto_embed=False, model=MockEncoder()
    )

    str1 = "0"
    str2 = "1"
    str3 = "2"

    ctx_str_1 = "ctx"
    ctx_str_2 = "ctx_"
    encoded_ctx_str_1 = "0:3.0 1:0.0"
    encoded_ctx_str_2 = "0:4.0 1:0.0"

    named_actions = {
        "action": rl_chain.EmbedAndKeep([{"a": str1, "b": str1}, str2, str3])
    }
    context = {
        "context1": rl_chain.EmbedAndKeep(ctx_str_1),
        "context2": rl_chain.EmbedAndKeep(ctx_str_2),
    }
    expected = "\n".join(
        [
            f"shared |context1_dense {encoded_ctx_str_1} |context2_dense {encoded_ctx_str_2} |context1_sparse raw:={ctx_str_1} |context2_sparse raw:={ctx_str_2}",
            f"0:-0.0:1.0 |a_dense 0:1.0 1:0.0 |b_dense 0:1.0 1:0.0 |a_sparse raw:=0 |b_sparse raw:=0",
            f"|action_dense 0:1.0 1:0.0 |action_sparse raw:=1",
            f"|action_dense 0:1.0 1:0.0 |action_sparse raw:=2",
        ]
    )  # noqa: E501

    selected = pick_best_chain.PickBestSelected(index=0, probability=1.0, score=0.0)
    event = pick_best_chain.PickBestEvent(
        inputs={}, to_select_from=named_actions, based_on=context, selected=selected
    )
    vw_ex_str = vw_cb_formatter(*featurizer.featurize(event))
    assert_vw_ex_equals(vw_ex_str, expected)


def test_pickbest_textembedder_more_namespaces_w_full_label_w_partial_emb() -> None:
    featurizer = pick_best_chain.PickBestFeaturizer(
        auto_embed=False, model=MockEncoder()
    )

    str1 = "0"
    str2 = "1"
    str3 = "2"

    ctx_str_1 = "ctx"
    ctx_str_2 = "ctx_"
    encoded_ctx_str_2 = "0:4.0 1:0.0"

    named_actions = {
        "action": [{"a": str1, "b": rl_chain.Embed(str1)}, str2, rl_chain.Embed(str3)]
    }
    context = {"context1": ctx_str_1, "context2": rl_chain.Embed(ctx_str_2)}

    expected = "\n".join(
        [
            f"shared |context2_dense {encoded_ctx_str_2} |context1_sparse raw:={ctx_str_1}",
            f"0:-0.0:1.0 |b_dense 0:1.0 1:0.0 |a_sparse raw:=0",
            f"|action_sparse raw:=1",
            f"|action_dense 0:1.0 1:0.0",
        ]
    )  # noqa: E501

    selected = pick_best_chain.PickBestSelected(index=0, probability=1.0, score=0.0)
    event = pick_best_chain.PickBestEvent(
        inputs={}, to_select_from=named_actions, based_on=context, selected=selected
    )
    vw_ex_str = vw_cb_formatter(*featurizer.featurize(event))
    assert_vw_ex_equals(vw_ex_str, expected)


def test_pickbest_textembedder_more_namespaces_w_full_label_w_partial_emakeep() -> None:
    featurizer = pick_best_chain.PickBestFeaturizer(
        auto_embed=False, model=MockEncoder()
    )

    str1 = "0"
    str2 = "1"
    str3 = "2"

    ctx_str_1 = "ctx"
    ctx_str_2 = "ctx_"
    encoded_ctx_str_2 = "0:4.0 1:0.0"

    named_actions = {
        "action": [
            {"a": str1, "b": rl_chain.EmbedAndKeep(str1)},
            str2,
            rl_chain.EmbedAndKeep(str3),
        ]
    }
    context = {"context1": ctx_str_1, "context2": rl_chain.EmbedAndKeep(ctx_str_2)}
    expected = "\n".join(
        [
            f"shared |context2_dense {encoded_ctx_str_2} |context1_sparse raw:={ctx_str_1} |context2_sparse raw:={ctx_str_2}",
            f"0:-0.0:1.0 |b_dense 0:1.0 1:0.0 |a_sparse raw:=0 |b_sparse raw:=0",
            f"|action_sparse raw:=1",
            f"|action_dense 0:1.0 1:0.0 |action_sparse raw:=2",
        ]
    )  # noqa: E501
    selected = pick_best_chain.PickBestSelected(index=0, probability=1.0, score=0.0)
    event = pick_best_chain.PickBestEvent(
        inputs={}, to_select_from=named_actions, based_on=context, selected=selected
    )
    vw_ex_str = vw_cb_formatter(*featurizer.featurize(event))
    assert_vw_ex_equals(vw_ex_str, expected)


def test_raw_features_underscored() -> None:
    featurizer = pick_best_chain.PickBestFeaturizer(
        auto_embed=False, model=MockEncoder()
    )
    str1 = "this is a long string"
    str1_underscored = str1.replace(" ", "_")
    encoded_str1 = f"0:{float(len(str1))} 1:0.0"

    ctx_str = "this is a long context"
    ctx_str_underscored = ctx_str.replace(" ", "_")
    encoded_ctx_str = f"0:{float(len(ctx_str))} 1:0.0"

    # No embeddings
    named_actions = {"action": [str1]}
    context = {"context": ctx_str}
    expected_no_embed = "\n".join(
        [
            f"shared |context_sparse raw:={ctx_str_underscored}",
            f"|action_sparse raw:={str1_underscored}",
        ]
    )

    event = pick_best_chain.PickBestEvent(
        inputs={}, to_select_from=named_actions, based_on=context
    )
    vw_ex_str = vw_cb_formatter(*featurizer.featurize(event))
    assert_vw_ex_equals(vw_ex_str, expected_no_embed)

    # Just embeddings
    named_actions = {"action": rl_chain.Embed([str1])}
    context = {"context": rl_chain.Embed(ctx_str)}
    expected_embed = "\n".join(
        [f"shared |context_dense {encoded_ctx_str}", f"|action_dense {encoded_str1}"]
    )
    event = pick_best_chain.PickBestEvent(
        inputs={}, to_select_from=named_actions, based_on=context
    )
    vw_ex_str = vw_cb_formatter(*featurizer.featurize(event))
    assert_vw_ex_equals(vw_ex_str, expected_embed)

    # Embeddings and raw features
    named_actions = {"action": rl_chain.EmbedAndKeep([str1])}
    context = {"context": rl_chain.EmbedAndKeep(ctx_str)}
    expected_embed_and_keep = "\n".join(
        [
            f"shared |context_dense {encoded_ctx_str} |context_sparse raw:={ctx_str_underscored}",
            f"|action_dense {encoded_str1} |action_sparse raw:={str1_underscored}",
        ]
    )  # noqa: E501
    event = pick_best_chain.PickBestEvent(
        inputs={}, to_select_from=named_actions, based_on=context
    )
    vw_ex_str = vw_cb_formatter(*featurizer.featurize(event))
    assert_vw_ex_equals(vw_ex_str, expected_embed_and_keep)
