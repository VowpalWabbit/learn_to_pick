from learn_to_pick.base import Event, Featurizer, Policy
from learn_to_pick.vw.model_repository import ModelRepository
from learn_to_pick.vw.logger import VwLogger
from typing import Any, List, Callable, TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    import vowpal_wabbit_next as vw

TEvent = TypeVar("TEvent", bound=Event)


def _parse_lines(parser: "vw.TextFormatParser", input_str: str) -> List["vw.Example"]:
    return [parser.parse_line(line) for line in input_str.split("\n")]


class VwPolicy(Policy):
    def __init__(
        self,
        model_repo: ModelRepository,
        vw_cmd: List[str],
        featurizer: Featurizer,
        formatter: Callable,
        vw_logger: VwLogger,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.model_repo = model_repo
        self.vw_cmd = vw_cmd
        self.workspace = self.model_repo.load(vw_cmd)
        self.featurizer = featurizer
        self.formatter = formatter
        self.vw_logger = vw_logger

    def format(self, event):
        return self.formatter(*self.featurizer.featurize(event))

    def predict(self, event: TEvent) -> Any:
        import vowpal_wabbit_next as vw

        text_parser = vw.TextFormatParser(self.workspace)
        return self.workspace.predict_one(_parse_lines(text_parser, self.format(event)))

    def learn(self, event: TEvent) -> None:
        import vowpal_wabbit_next as vw

        vw_ex = self.format(event)
        text_parser = vw.TextFormatParser(self.workspace)
        multi_ex = _parse_lines(text_parser, vw_ex)
        self.workspace.learn_one(multi_ex)

    def log(self, event: TEvent) -> None:
        if self.vw_logger.logging_enabled():
            vw_ex = self.format(event)
            self.vw_logger.log(vw_ex)

    def save(self) -> None:
        self.model_repo.save(self.workspace)
