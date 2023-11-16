from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Type, Union, Callable
from itertools import chain
import os
import numpy as np

from learn_to_pick import base

logger = logging.getLogger(__name__)

# sentinel object used to distinguish between
# user didn't supply anything or user explicitly supplied None
SENTINEL = object()


class PickBestSelected(base.Selected):
    index: Optional[int]
    probability: Optional[float]
    score: Optional[float]

    def __init__(
        self,
        index: Optional[int] = None,
        probability: Optional[float] = None,
        score: Optional[float] = None,
    ):
        self.index = index
        self.probability = probability
        self.score = score


class PickBestEvent(base.Event[PickBestSelected]):
    def __init__(
        self,
        inputs: Dict[str, Any],
        to_select_from: Dict[str, Any],
        based_on: Dict[str, Any],
        selected: Optional[PickBestSelected] = None,
    ):
        super().__init__(inputs=inputs, selected=selected or PickBestSelected())
        self.to_select_from = to_select_from
        self.based_on = based_on


class VwTxt:
    @staticmethod
    def _dense_2_str(values: base.DenseFeatures) -> str:
        return " ".join([f"{i}:{e}" for i, e in enumerate(values)])

    @staticmethod
    def _sparse_2_str(values: base.SparseFeatures) -> str:
        def _to_str(v):
            import numbers

            return f":{v}" if isinstance(v, numbers.Number) else f"={v}"

        return " ".join([f"{k}{_to_str(v)}" for k, v in values.items()])

    @staticmethod
    def featurized_2_str(obj: base.Featurized) -> str:
        return " ".join(
            chain.from_iterable(
                [
                    map(
                        lambda kv: f"|{kv[0]}_dense {VwTxt._dense_2_str(kv[1])}",
                        obj.dense.items(),
                    ),
                    map(
                        lambda kv: f"|{kv[0]}_sparse {VwTxt._sparse_2_str(kv[1])}",
                        obj.sparse.items(),
                    ),
                ]
            )
        )


class PickBestFeaturizer(base.Featurizer[PickBestEvent]):
    """
    Text Featurizer class that embeds the `BasedOn` and `ToSelectFrom` inputs into a format that can be used by the learning policy

    Attributes:
        model name (Any, optional): The type of embeddings to be used for feature representation. Defaults to BERT SentenceTransformer.
    """

    def __init__(
        self, auto_embed: bool, model: Optional[Any] = None, *args: Any, **kwargs: Any
    ):
        super().__init__(*args, **kwargs)

        if model is None:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer("all-mpnet-base-v2")

        self.model = model
        self.auto_embed = auto_embed

    def _dotproducts(self, context, actions):
        _context_dense = base.Featurized()
        for ns in context.sparse.keys():
            if "default_ft" in context.sparse[ns]:
                _context_dense[ns] = self.model.encode(context.sparse[ns]["default_ft"])

        _actions_dense = [base.Featurized() for _ in range(len(actions))]
        for _action, action in zip(_actions_dense, actions):
            for ns in action.sparse.keys():
                if "default_ft" in action.sparse[ns]:
                    _action[ns] = self.model.encode(action.sparse[ns]["default_ft"])

        context_names = list(_context_dense.dense.keys())
        context_matrix = np.stack(list(_context_dense.dense.values()))
        for _a, a in zip(_actions_dense, actions):
            action_names = list(_a.dense.keys())
            product = np.dot(context_matrix, np.stack(list(_a.dense.values())).T)
            a["dotprod"] = {
                f"{context_names[i]}_{action_names[j]}": product[i, j]
                for i in range(len(context_names))
                for j in range(len(action_names))
            }

    @staticmethod
    def _generic_namespace(featurized):
        result = base.SparseFeatures()
        for ns in featurized.sparse.keys():
            if "default_ft" in featurized.sparse[ns]:
                result[ns] = featurized.sparse[ns]["default_ft"]
        return result

    @staticmethod
    def _generic_namespaces(context, actions):
        context["@"] = PickBestFeaturizer._generic_namespace(context)
        for a in actions:
            a["#"] = PickBestFeaturizer._generic_namespace(a)

    def get_context_actions(self, event) -> Tuple[base.Featurized, List[base.Featurized]]:
        context = base.embed(event.based_on or {}, self.model)
        to_select_from_var_name, to_select_from = next(
            iter(event.to_select_from.items()), (None, None)
        )

        actions = (
            (
                base.embed(to_select_from, self.model, to_select_from_var_name)
                if event.to_select_from
                else None
            )
            if to_select_from
            else None
        )
        if not actions:
            raise ValueError(
                "Context and to_select_from must be provided in the inputs dictionary"
            )
        return context, actions

    def featurize(
        self, event: PickBestEvent
    ) -> Tuple[base.Featurized, List[base.Featurized], PickBestSelected]:
        context, actions = self.get_context_actions(event)

        if self.auto_embed:
            self._dotproducts(context, actions)
            PickBestFeaturizer._generic_namespaces(context, actions)

        return context, actions, event.selected


def vw_cb_formatter(
    context: base.Featurized, actions: List[base.Featurized], selected: PickBestSelected
) -> str:
    nactions = len(actions)
    context_str = f"shared {VwTxt.featurized_2_str(context)}"
    labels = ["" for _ in range(nactions)]
    if selected.score is not None:
        labels[
            selected.index
        ] = f"{selected.index}:{-selected.score}:{selected.probability} "
    actions_str = [f"{l}{VwTxt.featurized_2_str(a)}" for a, l in zip(actions, labels)]
    return "\n".join([context_str] + actions_str)


class PickBestRandomPolicy(base.Policy[PickBestEvent]):
    def predict(self, event: PickBestEvent) -> List[Tuple[int, float]]:
        num_items = len(event.to_select_from)
        return [(i, 1.0 / num_items) for i in range(num_items)]

    def learn(self, event: PickBestEvent) -> None:
        pass

    def log(self, event: PickBestEvent) -> None:
        pass


class PickBest(base.RLLoop[PickBestEvent]):
    """
    `PickBest` is a class designed to leverage a learned Policy for reinforcement learning with a context.

    Each invocation of the `run()` method should be equipped with a set of potential actions (`ToSelectFrom`) and will result in the selection of a specific action based on the `BasedOn` input.

    The standard operation flow of this run() call includes a loop:
        - The loop is invoked with inputs containing the `BasedOn` criteria and a list of potential actions (`ToSelectFrom`).
        - An action is selected based on the `BasedOn` input.
        - If a `selection_scorer` is provided, it is used to score the selection.
        - The internal Policy is updated with the `BasedOn` input, the chosen `ToSelectFrom` action, and the resulting score from the scorer.
        - The final pick is returned.

    Expected input dictionary format:
        - At least one variable encapsulated within `BasedOn` to serve as the selection criteria.
        - A single list variable within `ToSelectFrom`, representing potential actions for the learned Policy to pick from. This list can take the form of:
            - A list of strings, e.g., `action = ToSelectFrom(["action1", "action2", "action3"])`
            - A list of list of strings e.g. `action = ToSelectFrom([["action1", "another identifier of action1"], ["action2", "another identifier of action2"]])`
            - A list of dictionaries, where each dictionary represents an action with namespace names as keys and corresponding action strings as values. For instance, `action = ToSelectFrom([{"namespace1": ["action1", "another identifier of action1"], "namespace2": "action2"}, {"namespace1": "action3", "namespace2": "action4"}])`.

    Extends:
        RLLoop

    Attributes:
        featurizer (PickBestFeaturizer, optional): Is an advanced attribute. Responsible for embedding the `BasedOn` and `ToSelectFrom` inputs. If omitted, a default embedder is utilized.
    """

    def _call_before_predict(self, inputs: Dict[str, Any]) -> PickBestEvent:
        context, actions = base.get_based_on_and_to_select_from(inputs=inputs)
        if not actions:
            raise ValueError(
                "No variables using 'ToSelectFrom' found in the inputs. Please include at least one variable containing a list to select from."
            )

        if len(list(actions.values())) > 1:
            raise ValueError(
                "Only one variable using 'ToSelectFrom' can be provided in the inputs for PickBest run() call. Please provide only one variable containing a list to select from."
            )

        if not context:
            raise ValueError(
                "No variables using 'BasedOn' found in the inputs. Please include at least one variable containing information to base the selected of ToSelectFrom on."
            )

        event = PickBestEvent(inputs=inputs, to_select_from=actions, based_on=context)
        return event

    def _call_after_predict_before_scoring(
        self,
        inputs: Dict[str, Any],
        event: PickBestEvent,
        prediction: List[Tuple[int, float]],
    ) -> Tuple[Dict[str, Any], PickBestEvent]:
        import numpy as np

        prob_sum = sum(prob for _, prob in prediction)
        probabilities = [prob / prob_sum for _, prob in prediction]
        ## sample from the pmf
        sampled_index = np.random.choice(len(prediction), p=probabilities)
        sampled_ap = prediction[sampled_index]
        sampled_action = sampled_ap[0]
        sampled_prob = sampled_ap[1]
        event.selected = PickBestSelected(
            index=sampled_action, probability=sampled_prob
        )

        next_inputs = inputs.copy()

        picked = {}
        for k, v in event.to_select_from.items():
            picked[k] = v[event.selected.index]

        next_inputs[self.selected_based_on_input_key] = str(event.based_on)
        next_inputs[self.selected_input_key] = str(picked)

        return next_inputs, picked, event

    def _call_after_scoring_before_learning(
        self, event: PickBestEvent, score: Optional[float]
    ) -> PickBestEvent:
        if event.selected and score is not None:
            event.selected.score = score
        return event

    @classmethod
    def create(
        cls: Type[PickBest],
        policy: Optional[base.Policy] = None,
        llm=None,
        selection_scorer: Union[base.AutoSelectionScorer, object] = SENTINEL,
        **kwargs: Any,
    ) -> PickBest:
        if selection_scorer is SENTINEL and llm is None:
            raise ValueError("Either llm or selection_scorer must be provided")
        elif selection_scorer is SENTINEL:
            selection_scorer = base.AutoSelectionScorer(llm=llm)

        policy_args = {
            "featurizer": kwargs.pop("featurizer", None),
            "vw_cmd": kwargs.pop("vw_cmd", None),
            "model_save_dir": kwargs.pop("model_save_dir", None),
            "reset_model": kwargs.pop("reset_model", None),
            "rl_logs": kwargs.pop("rl_logs", None),
        }

        if policy and any(policy_args.values()):
            logger.warning(
                f"{[k for k, v in policy_args.items() if v]} will be ignored since nontrivial policy is provided, please set those arguments in the policy directly if needed"
            )

        if policy_args["model_save_dir"] is None:
            policy_args["model_save_dir"] = "./"
        if policy_args["reset_model"] is None:
            policy_args["reset_model"] = False

        return PickBest(
            policy=policy or PickBest.create_policy(**policy_args),
            selection_scorer=selection_scorer,
            **kwargs,
        )

    @staticmethod
    def create_policy(
        featurizer: Optional[base.Featurizer] = None,
        formatter: Optional[Callable] = None,
        vw_cmd: Optional[List[str]] = None,
        model_save_dir: str = "./",
        reset_model: bool = False,
        rl_logs: Optional[Union[str, os.PathLike]] = None,
    ):
        featurizer = featurizer or PickBestFeaturizer(auto_embed=False)
        formatter = formatter or vw_cb_formatter

        vw_cmd = vw_cmd or []
        interactions = []
        if vw_cmd:
            if "--cb_explore_adf" not in vw_cmd:
                raise ValueError(
                    "If vw_cmd is specified, it must include --cb_explore_adf"
                )
        else:
            interactions += ["--interactions=::"]
            vw_cmd = ["--cb_explore_adf", "--coin", "--squarecb", "--quiet"]

        if featurizer.auto_embed:
            interactions += [
                "--interactions=@#",
                "--ignore_linear=@",
                "--ignore_linear=#",
            ]

        vw_cmd = interactions + vw_cmd

        return base.VwPolicy(
            model_repo=base.ModelRepository(
                model_save_dir, with_history=True, reset=reset_model
            ),
            vw_cmd=vw_cmd,
            featurizer=featurizer,
            formatter=formatter,
            vw_logger=base.VwLogger(rl_logs),
        )

    def _default_policy(self):
        return PickBest.create_policy()
