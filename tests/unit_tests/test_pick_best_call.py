from typing import Any, Dict

import pytest
from test_utils import MockEncoder, MockEncoderReturnsList, assert_vw_ex_equals

import learn_to_pick
import learn_to_pick.base as rl_loop
from learn_to_pick.pick_best import vw_cb_formatter

encoded_keyword = "[encoded]"


class fake_llm_caller:
    def predict(self):
        return "hey"


class fake_llm_caller_with_score:
    def predict(self):
        return "3"


def test_multiple_ToSelectFrom_throws() -> None:
    pick = learn_to_pick.PickBest.create(
        llm=fake_llm_caller,
        featurizer=learn_to_pick.PickBestFeaturizer(
            auto_embed=False, model=MockEncoder()
        ),
    )
    actions = ["0", "1", "2"]
    with pytest.raises(ValueError):
        pick.run(
            User=learn_to_pick.BasedOn("Context"),
            action=learn_to_pick.ToSelectFrom(actions),
            another_action=learn_to_pick.ToSelectFrom(actions),
        )


def test_missing_basedOn_from_throws() -> None:
    pick = learn_to_pick.PickBest.create(
        llm=fake_llm_caller,
        featurizer=learn_to_pick.PickBestFeaturizer(
            auto_embed=False, model=MockEncoder()
        ),
    )
    actions = ["0", "1", "2"]
    with pytest.raises(ValueError):
        pick.run(action=learn_to_pick.ToSelectFrom(actions))


def test_ToSelectFrom_not_a_list_throws() -> None:
    pick = learn_to_pick.PickBest.create(
        llm=fake_llm_caller,
        featurizer=learn_to_pick.PickBestFeaturizer(
            auto_embed=False, model=MockEncoder()
        ),
    )
    actions = {"actions": ["0", "1", "2"]}
    with pytest.raises(ValueError):
        pick.run(
            User=learn_to_pick.BasedOn("Context"),
            action=learn_to_pick.ToSelectFrom(actions),
        )


def test_update_with_delayed_score_with_auto_validator_throws() -> None:
    auto_val_llm = fake_llm_caller_with_score
    pick = learn_to_pick.PickBest.create(
        llm=fake_llm_caller,
        selection_scorer=learn_to_pick.AutoSelectionScorer(llm=auto_val_llm),
        featurizer=learn_to_pick.PickBestFeaturizer(
            auto_embed=False, model=MockEncoder()
        ),
    )
    actions = ["0", "1", "2"]
    response = pick.run(
        User=learn_to_pick.BasedOn("Context"),
        action=learn_to_pick.ToSelectFrom(actions),
    )
    picked_metadata = response["picked_metadata"]  # type: ignore

    assert picked_metadata.selected.score == 3.0  # type: ignore
    with pytest.raises(RuntimeError):
        pick.update_with_delayed_score(
            chain_response=response, score=100  # type: ignore
        )


def test_update_with_delayed_score_force() -> None:
    # this LLM returns a number so that the auto validator will return that
    auto_val_llm = fake_llm_caller_with_score
    pick = learn_to_pick.PickBest.create(
        llm=fake_llm_caller,
        selection_scorer=learn_to_pick.AutoSelectionScorer(llm=auto_val_llm),
        featurizer=learn_to_pick.PickBestFeaturizer(
            auto_embed=False, model=MockEncoder()
        ),
    )
    actions = ["0", "1", "2"]
    response = pick.run(
        User=learn_to_pick.BasedOn("Context"),
        action=learn_to_pick.ToSelectFrom(actions),
    )
    picked_metadata = response["picked_metadata"]  # type: ignore
    assert picked_metadata.selected.score == 3.0  # type: ignore
    pick.update_with_delayed_score(
        chain_response=response, score=100, force_score=True  # type: ignore
    )
    assert picked_metadata.selected.score == 100.0  # type: ignore


def test_update_with_delayed_score() -> None:
    pick = learn_to_pick.PickBest.create(
        llm=fake_llm_caller,
        selection_scorer=None,
        featurizer=learn_to_pick.PickBestFeaturizer(
            auto_embed=False, model=MockEncoder()
        ),
    )
    actions = ["0", "1", "2"]
    response = pick.run(
        User=learn_to_pick.BasedOn("Context"),
        action=learn_to_pick.ToSelectFrom(actions),
    )
    picked_metadata = response["picked_metadata"]  # type: ignore
    assert picked_metadata.selected.score is None  # type: ignore
    pick.update_with_delayed_score(chain_response=response, score=100)  # type: ignore
    assert picked_metadata.selected.score == 100.0  # type: ignore


def test_user_defined_scorer() -> None:
    class CustomSelectionScorer(learn_to_pick.SelectionScorer):
        def score_response(
            self,
            inputs: Dict[str, Any],
            picked: Any,
            event: learn_to_pick.PickBestEvent,
        ) -> float:
            score = 200
            return score

    pick = learn_to_pick.PickBest.create(
        llm=fake_llm_caller,
        selection_scorer=CustomSelectionScorer(),
        featurizer=learn_to_pick.PickBestFeaturizer(
            auto_embed=False, model=MockEncoder()
        ),
    )
    actions = ["0", "1", "2"]
    response = pick.run(
        User=learn_to_pick.BasedOn("Context"),
        action=learn_to_pick.ToSelectFrom(actions),
    )
    picked_metadata = response["picked_metadata"]  # type: ignore
    assert picked_metadata.selected.score == 200.0  # type: ignore


def test_everything_embedded() -> None:
    featurizer = learn_to_pick.PickBestFeaturizer(auto_embed=False, model=MockEncoder())
    pick = learn_to_pick.PickBest.create(llm=fake_llm_caller, featurizer=featurizer)

    str1 = "0"
    str2 = "1"
    str3 = "2"
    action_dense = "0:1.0 1:0.0"

    ctx_str_1 = "context1"
    encoded_ctx_str_1 = "0:8.0 1:0.0"

    expected = "\n".join(
        [
            f"shared |User_dense {encoded_ctx_str_1} |User_sparse default_ft={ctx_str_1}",
            f"|action_dense {action_dense} |action_sparse default_ft={str1}",
            f"|action_dense {action_dense} |action_sparse default_ft={str2}",
            f"|action_dense {action_dense} |action_sparse default_ft={str3}",
        ]
    )  # noqa

    actions = [str1, str2, str3]

    response = pick.run(
        User=rl_loop.EmbedAndKeep(learn_to_pick.BasedOn(ctx_str_1)),
        action=rl_loop.EmbedAndKeep(learn_to_pick.ToSelectFrom(actions)),
    )
    picked_metadata = response["picked_metadata"]  # type: ignore
    vw_str = vw_cb_formatter(*featurizer.featurize(picked_metadata))  # type: ignore
    assert_vw_ex_equals(vw_str, expected)


def test_default_auto_embedder_is_off() -> None:
    featurizer = learn_to_pick.PickBestFeaturizer(auto_embed=False, model=MockEncoder())
    pick = learn_to_pick.PickBest.create(llm=fake_llm_caller, featurizer=featurizer)

    str1 = "0"
    str2 = "1"
    str3 = "2"
    ctx_str_1 = "context1"

    expected = "\n".join(
        [
            f"shared |User_sparse default_ft={ctx_str_1}",
            f"|action_sparse default_ft={str1}",
            f"|action_sparse default_ft={str2}",
            f"|action_sparse default_ft={str3}",
        ]
    )  # noqa

    actions = [str1, str2, str3]

    response = pick.run(
        User=learn_to_pick.base.BasedOn(ctx_str_1),
        action=learn_to_pick.base.ToSelectFrom(actions),
    )
    picked_metadata = response["picked_metadata"]  # type: ignore
    vw_str = vw_cb_formatter(*featurizer.featurize(picked_metadata))  # type: ignore
    assert_vw_ex_equals(vw_str, expected)


def test_default_w_embeddings_off() -> None:
    featurizer = learn_to_pick.PickBestFeaturizer(auto_embed=False, model=MockEncoder())
    pick = learn_to_pick.PickBest.create(llm=fake_llm_caller, featurizer=featurizer)

    str1 = "0"
    str2 = "1"
    str3 = "2"
    ctx_str_1 = "context1"

    expected = "\n".join(
        [
            f"shared |User_sparse default_ft={ctx_str_1}",
            f"|action_sparse default_ft={str1}",
            f"|action_sparse default_ft={str2}",
            f"|action_sparse default_ft={str3}",
        ]
    )  # noqa

    actions = [str1, str2, str3]

    response = pick.run(
        User=learn_to_pick.BasedOn(ctx_str_1),
        action=learn_to_pick.ToSelectFrom(actions),
    )
    picked_metadata = response["picked_metadata"]  # type: ignore
    vw_str = vw_cb_formatter(*featurizer.featurize(picked_metadata))  # type: ignore
    assert_vw_ex_equals(vw_str, expected)


def test_default_w_embeddings_on() -> None:
    featurizer = learn_to_pick.PickBestFeaturizer(
        auto_embed=True, model=MockEncoderReturnsList()
    )
    pick = learn_to_pick.PickBest.create(llm=fake_llm_caller, featurizer=featurizer)

    str1 = "0"
    str2 = "1"
    ctx_str_1 = "context1"
    dot_prod = "dotprod_sparse User_action:5.0"  # dot prod of [1.0, 2.0] and [1.0, 2.0]

    expected = "\n".join(
        [
            f"shared |User_sparse default_ft={ctx_str_1} |@_sparse User={ctx_str_1}",
            f"|action_sparse default_ft={str1} |{dot_prod} |#_sparse action={str1} ",
            f"|action_sparse default_ft={str2} |{dot_prod} |#_sparse action={str2} ",
        ]
    )  # noqa

    actions = [str1, str2]

    response = pick.run(
        User=learn_to_pick.BasedOn(ctx_str_1),
        action=learn_to_pick.ToSelectFrom(actions),
    )
    picked_metadata = response["picked_metadata"]  # type: ignore
    vw_str = vw_cb_formatter(*featurizer.featurize(picked_metadata))  # type: ignore
    assert_vw_ex_equals(vw_str, expected)


# TODO: fix behavior and test
# Right now expected value is: shared |User 0:1.0 1:2.0 |@ User=0:1.0 1:2.0 |User2 context2 |@ User2=context2
# While returned one is:shared |User 0:1.0 1:2.0 |User2 context2 |@ User=0:1.0 1:2.0 User2=context2
# But both doesn't make sense => auto + manual embedding scenario should be reconsidered
# And vw specific embedding representation should be removed from base.py to some vw-specific class

# def test_default_embeddings_mixed_w_explicit_user_embeddings() -> None:
#     featurizer = learn_to_pick.PickBestFeaturizer(
#         auto_embed=True, model=MockEncoderReturnsList()
#     )
#     pick = learn_to_pick.PickBest.create(llm=fake_llm_caller, featurizer=featurizer)

#     str1 = "0"
#     str2 = "1"
#     encoded_str2 = rl_loop._stringify_embedding([1.0, 2.0])
#     ctx_str_1 = "context1"
#     ctx_str_2 = "context2"
#     encoded_ctx_str_1 = rl_loop._stringify_embedding([1.0, 2.0])
#     dot_prod = "dotprod 0:5.0 1:5.0"  # dot prod of [1.0, 2.0] and [1.0, 2.0]

#     expected = f"""shared |User {encoded_ctx_str_1} |@ User={encoded_ctx_str_1} |User2 {ctx_str_2} |@ User2={ctx_str_2}\n|action {str1} |# action={str1} |{dot_prod}\n|action {encoded_str2} |# action={encoded_str2} |{dot_prod}"""  # noqa

#     actions = [str1, learn_to_pick.Embed(str2)]

#     response = pick.run(
#         User=learn_to_pick.BasedOn(learn_to_pick.Embed(ctx_str_1)),
#         User2=learn_to_pick.BasedOn(ctx_str_2),
#         action=learn_to_pick.ToSelectFrom(actions),
#     )
#     picked_metadata = response["picked_metadata"]  # type: ignore
#     vw_str = featurizer.format(picked_metadata)  # type: ignore
#     assert_vw_ex_equals(vw_str, expected)


def test_default_no_scorer_specified() -> None:
    pick = learn_to_pick.PickBest.create(
        llm=fake_llm_caller_with_score,
        featurizer=learn_to_pick.PickBestFeaturizer(
            auto_embed=False, model=MockEncoder()
        ),
    )
    response = pick.run(
        User=learn_to_pick.BasedOn("Context"),
        action=learn_to_pick.ToSelectFrom(["0", "1", "2"]),
    )
    # chain llm used for both basic prompt and for scoring
    picked_metadata = response["picked_metadata"]  # type: ignore
    assert picked_metadata.selected.score == 3.0  # type: ignore


def test_explicitly_no_scorer() -> None:
    pick = learn_to_pick.PickBest.create(
        llm=fake_llm_caller,
        selection_scorer=None,
        featurizer=learn_to_pick.PickBestFeaturizer(
            auto_embed=False, model=MockEncoder()
        ),
    )
    response = pick.run(
        User=learn_to_pick.BasedOn("Context"),
        action=learn_to_pick.ToSelectFrom(["0", "1", "2"]),
    )
    # chain llm used for both basic prompt and for scoring
    picked_metadata = response["picked_metadata"]  # type: ignore
    assert picked_metadata.selected.score is None  # type: ignore


def test_auto_scorer_with_user_defined_llm() -> None:
    scorer_llm = fake_llm_caller_with_score
    pick = learn_to_pick.PickBest.create(
        llm=fake_llm_caller,
        selection_scorer=learn_to_pick.AutoSelectionScorer(llm=scorer_llm),
        featurizer=learn_to_pick.PickBestFeaturizer(
            auto_embed=False, model=MockEncoder()
        ),
    )
    response = pick.run(
        User=learn_to_pick.BasedOn("Context"),
        action=learn_to_pick.ToSelectFrom(["0", "1", "2"]),
    )
    # chain llm used for both basic prompt and for scoring
    picked_metadata = response["picked_metadata"]  # type: ignore
    assert picked_metadata.selected.score == 3  # type: ignore


def test_activate_and_deactivate_scorer() -> None:
    llm = fake_llm_caller
    scorer_llm = fake_llm_caller_with_score
    pick = learn_to_pick.PickBest.create(
        llm=llm,
        selection_scorer=learn_to_pick.base.AutoSelectionScorer(llm=scorer_llm),
        featurizer=learn_to_pick.PickBestFeaturizer(
            auto_embed=False, model=MockEncoder()
        ),
    )
    response = pick.run(
        User=learn_to_pick.base.BasedOn("Context"),
        action=learn_to_pick.base.ToSelectFrom(["0", "1", "2"]),
    )
    # chain llm used for both basic prompt and for scoring
    picked_metadata = response["picked_metadata"]  # type: ignore
    assert picked_metadata.selected.score == 3  # type: ignore

    pick.deactivate_selection_scorer()
    response = pick.run(
        User=learn_to_pick.base.BasedOn("Context"),
        action=learn_to_pick.base.ToSelectFrom(["0", "1", "2"]),
    )
    picked_metadata = response["picked_metadata"]  # type: ignore
    assert picked_metadata.selected.score is None  # type: ignore

    pick.activate_selection_scorer()
    response = pick.run(
        User=learn_to_pick.base.BasedOn("Context"),
        action=learn_to_pick.base.ToSelectFrom(["0", "1", "2"]),
    )
    picked_metadata = response["picked_metadata"]  # type: ignore
    assert picked_metadata.selected.score == 3  # type: ignore
