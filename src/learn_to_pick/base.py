from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Generic,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

from learn_to_pick.metrics import MetricsTrackerAverage, MetricsTrackerRollingWindow

from learn_to_pick.features import Featurized, DenseFeatures, SparseFeatures
from enum import Enum

if TYPE_CHECKING:
    import vowpal_wabbit_next as vw

logger = logging.getLogger(__name__)


class Role(Enum):
    CONTEXT = 1
    ACTIONS = 2


class _Roled:
    def __init__(self, value: Any, role: Role):
        self.value = value
        self.role = role

    def __str__(self) -> str:
        return str(self.value)

    __repr__ = __str__


def BasedOn(anything: Any) -> _Roled:
    return _Roled(anything, Role.CONTEXT)


def ToSelectFrom(anything: Any) -> _Roled:
    if not isinstance(anything, list):
        raise ValueError("ToSelectFrom must be a list to select from")
    return _Roled(anything, Role.ACTIONS)


class _Input:
    def __init__(self, value: Any, keep: bool = True, embed: bool = False):
        self.value = value
        self.keep = keep
        self.embed = embed

    def __str__(self) -> str:
        return str(self.value)

    @staticmethod
    def create(value: Any, *args, **kwargs):
        if isinstance(value, _Roled):
            return _Roled(_Input.create(value.value, *args, **kwargs), value.role)
        if isinstance(value, list):
            return [_Input.create(v, *args, **kwargs) for v in value]
        if isinstance(value, dict):
            return {k: _Input.create(v, *args, **kwargs) for k, v in value.items()}
        if isinstance(value, _Input):  # should we swap? it will allow overwriting
            return value
        return _Input(value, *args, **kwargs)

    __repr__ = __str__


def Embed(anything: Any, keep: bool = False) -> Any:
    return _Input.create(anything, keep=keep, embed=True)


def EmbedAndKeep(anything: Any) -> Any:
    return Embed(anything, keep=True)


# helper functions


def filter_inputs(inputs: Dict[str, Any], role: Role) -> Dict[str, Any]:
    return {
        k: v.value
        for k, v in inputs.items()
        if isinstance(v, _Roled) and v.role == role
    }


# end helper functions


class Selected(ABC):
    pass


TSelected = TypeVar("TSelected", bound=Selected)


class Event(Generic[TSelected], ABC):
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    selected: Optional[TSelected]

    def __init__(self, inputs: Dict[str, Any], selected: Optional[TSelected] = None):
        self.inputs = inputs
        self.selected = selected


TEvent = TypeVar("TEvent", bound=Event)


class Policy(Generic[TEvent], ABC):
    def __init__(self, **kwargs: Any):
        pass

    @abstractmethod
    def predict(self, event: TEvent) -> Any:
        ...

    @abstractmethod
    def learn(self, event: TEvent) -> None:
        ...

    @abstractmethod
    def log(self, event: TEvent) -> None:
        ...

    def save(self) -> None:
        pass


class Featurizer(Generic[TEvent], ABC):
    def __init__(self, *args: Any, **kwargs: Any):
        pass

    @abstractmethod
    def featurize(self, event: TEvent) -> Any:
        ...


class SelectionScorer(Generic[TEvent], ABC):
    """
    Abstract method to grade the selection.
    Subclasses should implement the score_response method to define custom scoring logic.
    """

    @abstractmethod
    def score_response(self, inputs: Dict[str, Any], picked: Any, event: TEvent) -> Any:
        """
        Calculate and return the score for the selected response.

        This is an abstract method and should be implemented by subclasses.
        The method defines a blueprint for applying scoring logic based on the provided
        inputs, the selection made by the policy, and additional metadata from the event.

        Args:
            inputs (Dict[str, Any]): The inputs provided to the picker.
            picked (Any): The selection made by the policy.
            event (TEvent): Metadata associated with the selection event.

        Returns:
            The calculated score for the selected response.
        """
        ...


class AutoSelectionScorer(SelectionScorer[Event]):
    def __init__(
        self,
        llm,
        prompt: Union[Any, None] = None,
        scoring_criteria_template_str: Optional[str] = None,
    ):
        self.llm = llm
        self.prompt = prompt
        if prompt is None and scoring_criteria_template_str is None:
            self.prompt = AutoSelectionScorer.get_default_prompt()
        elif prompt is None and scoring_criteria_template_str is not None:
            default_system_prompt = AutoSelectionScorer.get_default_system_prompt()
            self.prompt = default_system_prompt + scoring_criteria_template_str

    @staticmethod
    def get_default_system_prompt() -> str:
        return """
            PLEASE RESPOND ONLY WITH A SINGLE FLOAT AND NO OTHER TEXT EXPLANATION\n You are a strict judge that is called on to rank a response based on given criteria. You must respond with your ranking by providing a single float within the range [0, 1], 0 being very bad response and 1 being very good response.
        """

    @staticmethod
    def get_default_prompt() -> str:
        human_template = """Given this based_on "{selected_based_on}" \
            as the most important attribute, rank how good or bad this text is: \
                "{picked}"."""
        default_system_prompt = AutoSelectionScorer.get_default_system_prompt()
        return default_system_prompt + human_template

    @staticmethod
    def format_with_ignoring_extra_args(prompt, inputs):
        import string

        # Extract placeholders from the prompt
        placeholders = [
            field[1] for field in string.Formatter().parse(str(prompt)) if field[1]
        ]

        # Keep only the inputs that have corresponding placeholders in the prompt
        relevant_inputs = {k: v for k, v in inputs.items() if k in placeholders}

        return prompt.format(**relevant_inputs)

    def score_response(
        self, inputs: Dict[str, Any], picked: Any, event: Event
    ) -> float:
        p = AutoSelectionScorer.format_with_ignoring_extra_args(self.prompt, inputs)
        ranking = self.llm.predict(p)
        ranking = ranking.strip()
        try:
            resp = float(ranking)
            return resp
        except Exception as e:
            raise RuntimeError(
                f"The auto selection scorer did not manage to score the response, there is always the option to try again or tweak the reward prompt. Error: {e}"
            )


class RLLoop(Generic[TEvent]):
    """
    The `RLLoop` class leverages a learned Policy for reinforcement learning.

    The standard operation flow of this run() call includes a loop:
            - The loop is invoked with inputs
            - A decision is made based on the inputs.
            - If a `selection_scorer` is provided, it is used to score the decision/selection.
            - The score is used to update the internal Policy.
            - The decision is returned.

    Attributes:
        - selection_scorer (Union[SelectionScorer, None]): Scorer for the selection. Can be set to None.
        - policy (Optional[Policy]): The policy used by the chain to learn to populate a dynamic prompt.
        - metrics (Optional[Union[MetricsTrackerRollingWindow, MetricsTrackerAverage]]): Tracker for metrics, can be set to None.

    Initialization Attributes:
        - featurizer (Featurizer): Featurizer used for the `BasedOn` and `ToSelectFrom` inputs.
        - model_save_dir (str, optional): Directory for saving the VW model. Default is the current directory.
        - reset_model (bool): If set to True, the model starts training from scratch. Default is False.
        - vw_cmd (List[str], optional): Command line arguments for the VW model.
        - policy (Type[VwPolicy]): Policy used by the chain.
        - rl_logs (Optional[Union[str, os.PathLike]]): Path for the VW logs.
        - metrics_step (int): Step for the metrics tracker. Default is -1. If set without metrics_window_size, average metrics will be tracked, otherwise rolling window metrics will be tracked.
        - metrics_window_size (int): Window size for the metrics tracker. Default is -1. If set, rolling window metrics will be tracked.

    Notes:
        By default the class initializes the VW model using the provided arguments. If `selection_scorer` is not provided, a warning is logged, indicating that no reinforcement learning will occur unless the `update_with_delayed_score` method is called.
    """

    # Define the default values as class attributes
    selected_based_on_input_key = "selected_based_on"
    selected_input_key = "picked"

    def __init__(
        self,
        policy: Optional[Policy] = None,
        selection_scorer: Union[SelectionScorer, None] = None,
        selection_scorer_activated: bool = True,
        metrics_step: int = -1,
        metrics_window_size: int = -1,
        callbacks_before_scoring: list = [],
    ):
        self.selection_scorer = selection_scorer
        self.policy = policy or self._default_policy()
        self.selection_scorer_activated = selection_scorer_activated
        self.metrics_step = metrics_step
        self.metrics_window_size = metrics_window_size
        self.callbacks_before_scoring = callbacks_before_scoring

        if self.selection_scorer is None:
            logger.warning(
                "No selection scorer provided, which means that no reinforcement learning will be done in the RL chain unless update_with_delayed_score is called."
            )

        if metrics_window_size > 0:
            self.metrics = MetricsTrackerRollingWindow(
                step=metrics_step, window_size=metrics_window_size
            )
        else:
            self.metrics = MetricsTrackerAverage(step=metrics_step)

    @abstractmethod
    def _default_policy(self):
        ...

    def update_with_delayed_score(
        self, score: float, chain_response: Dict[str, Any], force_score: bool = False
    ) -> None:
        """
        Updates the learned policy with the score provided.
        Will raise an error if selection_scorer is set, and force_score=True was not provided during the method call
        """
        if self._can_use_selection_scorer() and not force_score:
            raise RuntimeError(
                "The selection scorer is set, and force_score was not set to True. Please set force_score=True to use this function."
            )
        if self.metrics:
            self.metrics.on_feedback(score)
        event: TEvent = chain_response["picked_metadata"]
        self._call_after_scoring_before_learning(event=event, score=score)
        self.policy.learn(event=event)
        self.policy.log(event=event)

    def deactivate_selection_scorer(self) -> None:
        """
        Deactivates the selection scorer, meaning that there will be no automatic scoring, and the policy will not be automatically updated.
        """
        self.selection_scorer_activated = False

    def activate_selection_scorer(self) -> None:
        """
        Activates the selection scorer, meaning that scoring happens automatically and the policy is updated automatically.
        """
        self.selection_scorer_activated = True

    def save_progress(self) -> None:
        """
        This function should be called to save the state of the learned policy model.
        """
        self.policy.save()

    def _can_use_selection_scorer(self) -> bool:
        """
        Returns whether the chain can use the selection scorer to score responses or not.
        """
        return self.selection_scorer is not None and self.selection_scorer_activated

    @abstractmethod
    def _call_before_predict(self, inputs: Dict[str, Any]) -> TEvent:
        ...

    @abstractmethod
    def _call_after_predict_before_scoring(
        self, inputs: Dict[str, Any], event: Event, prediction: List[Tuple[int, float]]
    ) -> Tuple[Dict[str, Any], Event]:
        ...

    def _call_after_scoring_before_learning(
        self, event: Event, score: Optional[float]
    ) -> Event:
        ...

    def run(self, *args, **kwargs) -> Dict[str, Any]:
        """
        The standard operation flow of this run() call includes a loop:
            - The loop is invoked with inputs
            - A decision is made based on the inputs.
            - If a `selection_scorer` is provided, it is used to score the decision/selection.
            - The score is used to update the internal Policy.
            - The decision is returned.
        """
        if args and not kwargs:
            inputs = args[0]
        elif kwargs and not args:
            inputs = kwargs
        else:
            raise ValueError(
                "Either a dictionary positional argument or keyword arguments should be provided"
            )

        if self.selected_based_on_input_key in inputs:
            raise ValueError(
                f"The input key {self.selected_based_on_input_key} is reserved. Please use a different key."
            )

        if self.selected_input_key in inputs:
            raise ValueError(
                f"The input key {self.selected_input_key} is reserved. Please use a different key."
            )

        event: TEvent = self._call_before_predict(inputs=inputs)
        prediction = self.policy.predict(event=event)
        if self.metrics:
            self.metrics.on_decision()

        next_inputs, picked, event = self._call_after_predict_before_scoring(
            inputs=inputs, event=event, prediction=prediction
        )

        for callback_func in self.callbacks_before_scoring:
            try:
                next_inputs, event = callback_func(
                    inputs=next_inputs, picked=picked, event=event
                )
            except Exception as e:
                logger.info(f"Callback function {callback_func} failed, error: {e}")

        score = None
        try:
            if self._can_use_selection_scorer():
                score = self.selection_scorer.score_response(
                    inputs=next_inputs, picked=picked, event=event
                )
        except Exception as e:
            logger.info(
                f"The selection scorer was not able to score, and the chain was not able to adjust to this response, error: {e}"
            )

        event = self._call_after_scoring_before_learning(score=score, event=event)

        if self.metrics and event.selected.score is not None:
            self.metrics.on_feedback(event.selected.score)
        self.policy.learn(event=event)
        self.policy.log(event=event)

        event.outputs = next_inputs
        return {"picked": picked, "picked_metadata": event}


def _embed_string_type(
    item: Union[str, _Input], model: Any, namespace: str
) -> Featurized:
    """Helper function to embed a string or an _Embed object."""
    import re

    result = Featurized()
    if isinstance(item, _Input):
        if item.embed:
            result[namespace] = DenseFeatures(model.encode(item.value))
        if item.keep:
            keep_str = item.value.replace(" ", "_")
            result[namespace] = {"default_ft": re.sub(r"[\t\n\r\f\v]+", " ", keep_str)}
    elif isinstance(item, str):
        encoded = item.replace(" ", "_")
        result[namespace] = {"default_ft": re.sub(r"[\t\n\r\f\v]+", " ", encoded)}
    else:
        raise ValueError(f"Unsupported type {type(item)} for embedding")

    return result


def _embed_dict_type(item: Dict, model: Any) -> Featurized:
    """Helper function to embed a dictionary item."""
    result = Featurized()
    for ns, embed_item in item.items():
        if isinstance(embed_item, list):
            for idx, embed_list_item in enumerate(embed_item):
                result.merge(_embed_string_type(embed_list_item, model, f"{ns}_{idx}"))
        else:
            result.merge(_embed_string_type(embed_item, model, ns))
    return result


def _embed_list_type(
    item: list, model: Any, namespace: Optional[str] = None
) -> List[Featurized]:
    result = []
    for embed_item in item:
        if isinstance(embed_item, dict):
            result.append(_embed_dict_type(embed_item, model))
        elif isinstance(embed_item, list):
            result.append(Featurized())
            for idx, embed_list_item in enumerate(embed_item):
                result[-1].merge(_embed_string_type(embed_list_item, model, f"{idx}"))
        else:
            result.append(_embed_string_type(embed_item, model, namespace))
    return result


def embed(
    to_embed: Union[Union[str, _Input], Dict, List[Union[str, _Input]], List[Dict]],
    model: Any,
    namespace: Optional[str] = None,
) -> Union[Featurized, List[Featurized]]:
    """
    Embeds the actions or context using the SentenceTransformer model (or a model that has an `encode` function)

    Attributes:
        to_embed: (Union[Union(str, _Embed(str)), Dict, List[Union(str, _Embed(str))], List[Dict]], required) The text to be embedded, either a string, a list of strings or a dictionary or a list of dictionaries.
        namespace: (str, optional) The default namespace to use when dictionary or list of dictionaries not provided.
        model: (Any, required) The model to use for embedding
    Returns:
        List[Dict[str, str]]: A list of dictionaries where each dictionary has the namespace as the key and the embedded string as the value
    """
    if (isinstance(to_embed, _Input) and isinstance(to_embed.value, str)) or isinstance(
        to_embed, str
    ):
        return _embed_string_type(to_embed, model, namespace)
    elif isinstance(to_embed, dict):
        return _embed_dict_type(to_embed, model)
    elif isinstance(to_embed, list):
        return _embed_list_type(to_embed, model, namespace)
    else:
        raise ValueError("Invalid input format for embedding")
