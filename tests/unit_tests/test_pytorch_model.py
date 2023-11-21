import random
import torch
import os
import pytest
import shutil

import learn_to_pick


CHECKPOINT_DIR = "test_models"


@pytest.fixture
def remove_checkpoint():
    yield
    if os.path.isdir(CHECKPOINT_DIR):
        shutil.rmtree(CHECKPOINT_DIR)


class CustomSelectionScorer(learn_to_pick.SelectionScorer):
    def get_score(self, user, time_of_day, article):
        preferences = {
            "Tom": {"morning": "politics", "afternoon": "music"},
            "Anna": {"morning": "sports", "afternoon": "politics"},
        }

        return int(preferences[user][time_of_day] == article)

    def score_response(
        self, inputs, picked, event: learn_to_pick.PickBestEvent
    ) -> float:
        chosen_article = picked["article"]
        user = event.based_on["user"]
        time_of_day = event.based_on["time_of_day"]
        score = self.get_score(user, time_of_day, chosen_article)
        return score


class Simulator:
    def __init__(self, seed=7492381):
        self.random = random.Random(seed)
        self.users = ["Tom", "Anna"]
        self.times_of_day = ["morning", "afternoon"]
        self.articles = ["politics", "sports", "music"]

    def _choose_user(self):
        return self.random.choice(self.users)

    def _choose_time_of_day(self):
        return self.random.choice(self.times_of_day)

    def run(self, pytorch_picker, T):
        for i in range(T):
            user = self._choose_user()
            time_of_day = self._choose_time_of_day()
            pytorch_picker.run(
                article=learn_to_pick.ToSelectFrom(self.articles),
                user=learn_to_pick.BasedOn(user),
                time_of_day=learn_to_pick.BasedOn(time_of_day),
            )


def verify_same_models(model1, model2):
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        assert torch.equal(p1, p2), "The models' parameters are not equal."

    for (name1, buffer1), (name2, buffer2) in zip(
        model1.named_buffers(), model2.named_buffers()
    ):
        assert name1 == name2, "Buffer names do not match."
        assert torch.equal(buffer1, buffer2), f"The buffers {name1} are not equal."


def verify_same_optimizers(optimizer1, optimizer2):
    if type(optimizer1) != type(optimizer2):
        return False

    if optimizer1.defaults != optimizer2.defaults:
        return False

    state_dict1 = optimizer1.state_dict()
    state_dict2 = optimizer2.state_dict()

    if state_dict1.keys() != state_dict2.keys():
        return False

    for key in state_dict1:
        if key == "state":
            if state_dict1[key].keys() != state_dict2[key].keys():
                return False
            for subkey in state_dict1[key]:
                if not torch.equal(state_dict1[key][subkey], state_dict2[key][subkey]):
                    return False
        else:
            if state_dict1[key] != state_dict2[key]:
                return False

    return True


def test_save_load(remove_checkpoint):
    sim1 = Simulator()
    sim2 = Simulator()

    fe = learn_to_pick.PyTorchFeaturizer(auto_embed=True)
    first_model_path = f"{CHECKPOINT_DIR}/first.checkpoint"

    torch.manual_seed(0)
    first_byom = learn_to_pick.PyTorchPolicy(feature_embedder=fe)
    second_byom = learn_to_pick.PyTorchPolicy(feature_embedder=fe)

    torch.manual_seed(0)

    first_picker = learn_to_pick.PickBest.create(
        policy=first_byom, selection_scorer=CustomSelectionScorer()
    )
    sim1.run(first_picker, 5)
    first_byom.save(first_model_path)

    second_byom.load(first_model_path)
    second_picker = learn_to_pick.PickBest.create(
        policy=second_byom, selection_scorer=CustomSelectionScorer()
    )
    sim1.run(second_picker, 5)

    torch.manual_seed(0)
    all_byom = learn_to_pick.PyTorchPolicy(feature_embedder=fe)
    torch.manual_seed(0)
    all_picker = learn_to_pick.PickBest.create(
        policy=all_byom, selection_scorer=CustomSelectionScorer()
    )
    sim2.run(all_picker, 10)

    verify_same_models(second_byom.workspace, all_byom.workspace)
    verify_same_optimizers(second_byom.workspace.optim, all_byom.workspace.optim)
