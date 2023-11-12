from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Type, Union, Iterable
from itertools import chain
import os

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
        super().__init__(inputs=inputs, selected=selected)
        self.to_select_from = to_select_from
        self.based_on = based_on


class VwTxt:
    @staticmethod
    def embedding(embedding: List[float]) -> str:
        return " ".join([f"{i}:{e}" for i, e in enumerate(embedding)])

    @staticmethod
    def features(features: Union[str, List[str]]) -> str:
        return " ".join(features) if isinstance(features, list) else features

    @staticmethod
    def _namespaces(ns: Iterable[Tuple[str, Union[str, List[str]]]]):
        return " ".join(f"|{k} {VwTxt.features(v)}" for k, v in ns)

    @staticmethod
    def ns(ns: Union[Iterable[Tuple[str, Any]], List[Dict[str, Any]], Dict[str, Any]]):
        if isinstance(ns, List):
            ns = chain.from_iterable(map(dict.items, ns))
        if isinstance(ns, Dict):
            ns = ns.items()
        return VwTxt._namespaces(ns)


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

    def get_label(self, event: PickBestEvent) -> tuple:
        cost = None
        if event.selected:
            chosen_action = event.selected.index
            cost = (
                -1.0 * event.selected.score
                if event.selected.score is not None
                else None
            )
            prob = event.selected.probability
            return chosen_action, cost, prob
        else:
            return None, None, None

    def get_context_and_action_embeddings(self, event: PickBestEvent) -> tuple:
        context_emb = base.embed(event.based_on, self.model) if event.based_on else None
        to_select_from_var_name, to_select_from = next(
            iter(event.to_select_from.items()), (None, None)
        )

        action_embs = (
            (
                base.embed(to_select_from, self.model, to_select_from_var_name)
                if event.to_select_from
                else None
            )
            if to_select_from
            else None
        )

        if not context_emb or not action_embs:
            raise ValueError(
                "Context and to_select_from must be provided in the inputs dictionary"
            )
        return context_emb, action_embs

    def get_indexed_dot_product(self, context_emb: List, action_embs: List) -> Dict:
        import numpy as np

        unique_contexts = set()
        for context_item in context_emb:
            for ns, ee in context_item.items():
                if isinstance(ee, list):
                    for ea in ee:
                        unique_contexts.add(f"{ns}={ea}")
                else:
                    unique_contexts.add(f"{ns}={ee}")

        encoded_contexts = self.model.encode(list(unique_contexts))
        context_embeddings = dict(zip(unique_contexts, encoded_contexts))

        unique_actions = set()
        for action in action_embs:
            for ns, e in action.items():
                if isinstance(e, list):
                    for ea in e:
                        unique_actions.add(f"{ns}={ea}")
                else:
                    unique_actions.add(f"{ns}={e}")

        encoded_actions = self.model.encode(list(unique_actions))
        action_embeddings = dict(zip(unique_actions, encoded_actions))

        action_matrix = np.stack([v for k, v in action_embeddings.items()])
        context_matrix = np.stack([v for k, v in context_embeddings.items()])
        dot_product_matrix = np.dot(context_matrix, action_matrix.T)

        indexed_dot_product: Dict = {}

        for i, context_key in enumerate(context_embeddings.keys()):
            indexed_dot_product[context_key] = {}
            for j, action_key in enumerate(action_embeddings.keys()):
                indexed_dot_product[context_key][action_key] = dot_product_matrix[i, j]

        return indexed_dot_product

    def format_auto_embed_on(self, event: PickBestEvent) -> str:
        chosen_action, cost, prob = self.get_label(event)
        context_emb, action_embs = self.get_context_and_action_embeddings(event)
        indexed_dot_product = self.get_indexed_dot_product(context_emb, action_embs)

        nactions = len(action_embs)

        def _tolist(v):
            return v if isinstance(v, list) else [v]

        labels = ["" for _ in range(nactions)]
        if cost is not None:
            labels[chosen_action] = f"{chosen_action}:{cost}:{prob} "

        dotprods = [{} for _ in range(nactions)]
        for i, action in enumerate(action_embs):
            action["#"] = [f"{k}={v}" for k, _v in action.items() for v in _tolist(_v)]
            dotprods[i] = [
                v[f] for v in indexed_dot_product.values() for f in action["#"]
            ]

        actions_str = [
            f"{l}{VwTxt.ns(a)} |dotprod {VwTxt.embedding(dp)}"
            for l, a, dp in zip(labels, action_embs, dotprods)
        ]

        for item in context_emb:
            item["@"] = [f"{k}={v}" for k, _v in item.items() for v in _tolist(_v)]
        shared_str = f"shared {VwTxt.ns(context_emb)}"

        return "\n".join([shared_str] + actions_str)

    def format_auto_embed_off(self, event: PickBestEvent) -> str:
        """
        Converts the `BasedOn` and `ToSelectFrom` into a format that can be used by VW
        """
        chosen_action, cost, prob = self.get_label(event)
        context_emb, action_embs = self.get_context_and_action_embeddings(event)
        nactions = len(action_embs)

        context_str = f"shared {VwTxt.ns(context_emb)}"

        labels = ["" for _ in range(nactions)]
        if cost is not None:
            labels[chosen_action] = f"{chosen_action}:{cost}:{prob} "
        actions_str = [f"{l}{VwTxt.ns(a)}" for a, l in zip(action_embs, labels)]
        return "\n".join([context_str] + actions_str)

    def format(self, event: PickBestEvent) -> str:
        if self.auto_embed:
            return self.format_auto_embed_on(event)
        else:
            return self.format_auto_embed_off(event)


class PickBestRandomPolicy(base.Policy[PickBestEvent]):
    def __init__(self):
        ...

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
        selected = PickBestSelected(index=sampled_action, probability=sampled_prob)
        event.selected = selected

        next_inputs = inputs.copy()

        # only one key, value pair in event.to_select_from
        value = next(iter(event.to_select_from.values()))
        v = (
            value[event.selected.index]
            if event.selected
            else event.to_select_from.values()
        )

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
        vw_cmd: Optional[List[str]] = None,
        model_save_dir: str = "./",
        reset_model: bool = False,
        rl_logs: Optional[Union[str, os.PathLike]] = None,
    ):
        if not featurizer:
            featurizer = PickBestFeaturizer(auto_embed=False)

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
            vw_logger=base.VwLogger(rl_logs),
        )

    def _default_policy(self):
        return PickBest.create_policy()
