import learn_to_pick as rl_chain
from sentence_transformers import SentenceTransformer
import torch

class PyTorchFeatureEmbedder(): #rl_chain.Embedder[rl_chain.PickBestEvent]
    def __init__(
        self, auto_embed, model = None, *args, **kwargs
    ):
        if model is None:
            model = model = SentenceTransformer('all-MiniLM-L6-v2')

        self.model = model
        self.auto_embed = auto_embed

    def encode(self, stuff):
        embeddings = self.model.encode(stuff, convert_to_tensor=True)
        normalized = torch.nn.functional.normalize(embeddings)
        return normalized

    def get_label(self, event: rl_chain.PickBestEvent) -> tuple:
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

    def get_context_and_action_embeddings(self, event: rl_chain.PickBestEvent) -> tuple:
        context_emb = rl_chain.embed(event.based_on, self) if event.based_on else None
        to_select_from_var_name, to_select_from = next(
            iter(event.to_select_from.items()), (None, None)
        )

        action_embs = (
            (
                rl_chain.embed(to_select_from, self, to_select_from_var_name)
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

    def format(self, event: rl_chain.PickBestEvent):
        chosen_action, cost, prob = self.get_label(event)
        context_emb, action_embs = self.get_context_and_action_embeddings(event)

        context = ""
        for context_item in context_emb:
            for ns, based_on in context_item.items():
                e = " ".join(based_on) if isinstance(based_on, list) else based_on
                context += f"{ns}={e} "

        if self.auto_embed:
            context = self.encode([context])

        actions = []
        for action in action_embs:
            action_str = ""
            for ns, action_embedding in action.items():
                e = (
                    " ".join(action_embedding)
                    if isinstance(action_embedding, list)
                    else action_embedding
                )
                action_str += f"{ns}={e} "
            actions.append(action_str)

        if self.auto_embed:
            actions = self.encode(actions).unsqueeze(0)

        if cost is None:
            return context, actions
        else:
            return torch.Tensor([[-1.0 * cost]]), context, actions[:,chosen_action,:].unsqueeze(1)
