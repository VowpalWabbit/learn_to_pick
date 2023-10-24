## Introduction

`learn_to_pick` is a versatile Python library crafted for online learning in Reinforcement Learning (RL) loops. Specifically designed for Contextual Bandits, it allows you to choose an action from a set of possibilities based on certain criteria. After making a decision, you have multiple ways to evaluate its effectiveness and provide feedback back into the system, so that it can do better next time:

- Let a learned model (llm) determine the quality of the decision and provide a score.
- Use a custom score function to grade the decision.
- Directly specify the score manually and asynchronously.

The beauty of `learn_to_pick` is its flexibility. Whether you're a fan of VowpalWabbit or prefer PyTorch (coming soon), the library can seamlessly integrate with both, allowing them to be the brain behind your decisions.

## Installation

`pip install .`

## Usage

The `PickBest` scenario should be used when:

- Multiple options are available for a given scenario
- Only one option is optimal for a specific criteria or context
- There exists a mechanism to provide feedback on the suitability of the chosen option for the specific criteria

### Setting Up

Example usage with llm default scorer:

```python
from learn_to_pick import PickBest

# with an llm scorer

class fake_llm:
    def predict(self, inputs):
        print(f"here are the inputs: {inputs}")
        dummy_score = 1
        return dummy_score

picker = PickBest.create(llm=fake_llm())
result = picker.run(pick = learn_to_pick.ToSelectFrom(["option1", "option2"]),
                    criteria = learn_to_pick.BasedOn("some criteria")
)

print(result["picked"])
```

Example usage with custom scorer:

```python
from learn_to_pick import PickBest

# with a custom scorer

class CustomSelectionScorer(learn_to_pick.SelectionScorer):
    def score_response(self, inputs, picked, event) -> float:
        print(f"inputs: {inputs}")
        pick = picked[0]["pick"]
        criteria = event.based_on["criteria"]
        # evaluate pick based on criteria
        dummy_score = 1
        return dummy_score

picker = PickBest.create(selection_scorer=CustomSelectionScorer())
result = picker.run(pick = learn_to_pick.ToSelectFrom(["option1", "option2"]),
                    criteria = learn_to_pick.BasedOn("some criteria")
)

print(result["picked"])
```

```python
from learn_to_pick import PickBest

# with delayed score

picker = PickBest.create(selection_scorer=None)
result = picker.run(pick = learn_to_pick.ToSelectFrom(["option1", "option2"]),
                    criteria = learn_to_pick.BasedOn("some criteria")
)

print(result["picked"])

# evaluated the result asynchronusly in a different part of my system and determined a score
dummy_score = 1
picker.update_with_delayed_score(dummy_score, result)
```

`PickBest` is highly configurable to work with a VowpalWabbit decision making policy, a PyTorch decision making policy (coming soon), or with a custom user defined decision making policy

The main thing that needs to be decided from the get-go is:

- What is the action space, what are the options I have to select from
- Each time I need to pick an action from the action space, what are the criteria to make that decision
- What is my feedback mechanism

For the feedback mechanism there are three available options:

- Default: use a LLM to score the selection based on the criteria
  - works with a default scoring prompt if an llm object with a `predict()` call is passed in at initialization
  - custom scoring prompt can be provided
- Custom scoring function that has some logic baked in to evaluate selection based on the criteria
- Delayed explicit scoring, where a score is passed back into the module

In all three cases, when a score is calculated or provided, the decision making policy is updated in order to do better next time.


## Example Notebooks

- `demo.ipynb` mostly showcases basic usage
- `news_recommendation.ipynb` showcases a personalization scenario where we have to pick articles for specific users