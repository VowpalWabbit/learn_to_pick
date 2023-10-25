# learn to pick

- [Introduction](#introduction)
- [Installation](#installation)
- [Basic Usage](#basic-usage)
- [Example Notebooks](#example-notebooks)
- [Advanced Usage](#advanced-usage)
  - [Custom or Auto Scorer](#custom-or-auto-scorer)
  - [Register callback functions after decision and before scoring](#register-callback-functions-after-decision-and-before-scoring)
  - [Store progress of learned policy](#store-progress-of-learned-policy)
  - [Stop learning of learned policy](#stop-learning-of-learned-policy)
  - [Set a different policy](#set-a-different-policy)
  - [Different exploration algorithms and options for the default learned policy](#different-exploration-algorithms-and-options-for-the-default-learned-policy)
  - [Learned policy's data logs](#learned-policys-data-logs)
  - [Advanced featurization options](#advanced-featurization-options)
    - [auto_embed](#auto_embed)
    - [explicitly defined embeddings](#explicitly-defined-embeddings)
    - [custom featurization](#custom-featurization)
    - [other advanced featurization options](#other-advanced-featurization-options)


Note: all code examples presented here can be found in `notebooks/readme.ipynb`

## Introduction

`learn_to_pick` is a versatile Python library crafted for online learning in Reinforcement Learning (RL) loops. Specifically designed for Contextual Bandits, it allows you to choose an action from a set of possibilities based on certain criteria. After making a decision, you have multiple ways to evaluate its effectiveness and provide feedback back into the system, so that it can do better next time:

- Let a learned model (llm) determine the quality of the decision and provide a score.
- Use a custom score function to grade the decision.
- Directly specify the score manually and asynchronously.

The beauty of `learn_to_pick` is its flexibility. Whether you're a fan of VowpalWabbit or prefer PyTorch (coming soon), the library can seamlessly integrate with both, allowing them to be the brain behind your decisions.

## Installation

`pip install learn-to-pick`

## Basic Usage

The `PickBest` scenario should be used when:

- Multiple options are available for a given scenario
- Only one option is optimal for a specific criteria or context
- There exists a mechanism to provide feedback on the suitability of the chosen option for the specific criteria

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

The picker will:

- Make a selection using the decision making policy
- Call the scorer to evaluate the decision
- Update the decision making policy with the score

Note: callback functions can be registered at create time and they will be called after the decision has been made and before the scorer is called (see advanced section and `prompt_variable_injection.ipynb` for example usage)

Example usage with custom scorer:

```python
from learn_to_pick import PickBest

# with a custom scorer

class CustomSelectionScorer(learn_to_pick.SelectionScorer):
    def score_response(self, inputs, picked, event) -> float:
        print(f"inputs: {inputs}")
        pick = picked["pick"]
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

- `readme.ipynb` showcases all examples shown in this README
- `news_recommendation.ipynb` showcases a personalization scenario where we have to pick articles for specific users
- `prompt_variable_injection.ipynb` showcases learned prompt variable injection and registering callback functionality

### Advanced Usage

#### Custom or Auto Scorer

It is very important to get the selection scorer right since the policy uses it to learn. It determines what is called the reward in reinforcement learning, and more specifically in our Contextual Bandits setting.

The general advice is to keep the score between [0, 1], 0 being the worst selection, 1 being the best selection from the available `ToSelectFrom` variables, based on the `BasedOn` variables, but should be adjusted if the need arises.

In the examples provided above, the AutoSelectionScorer is set mostly to get users started but in real world scenarios it will most likely not be an adequate scorer function.

Auto scorer with custom scoring criteria prompt:

```python
scoring_criteria_template = "Given {criteria} rank how good or bad this selection is {pick}"
picker = learn_to_pick.PickBest.create(
    selection_scorer=learn_to_pick.AutoSelectionScorer(llm=llm, scoring_criteria_template_str=scoring_criteria_template),
)
```

AutoSelectionScorer will have a system prompt that makes sure that the llm responsible for scoring returns a single float.

However, if needed, a FULL scoring prompt can also be provided:

```python
# I want the score to be in the range [-1, 1] instead of the default [0, 1]

REWARD_PROMPT = """

Given {criteria} rank how good or bad this selection is {pick}

IMPORANT: you MUST return a single number between -1 and 1, -1 being bad, 1 being good

"""
picker = learn_to_pick.PickBest.create(
    selection_scorer=learn_to_pick.AutoSelectionScorer(llm=llm, prompt=REWARD_PROMPT)
)
```

Custom Scorer needs to extend the internal `SelectionScorer` and implement the `score_response` function

```python
class CustomSelectionScorer(learn_to_pick.SelectionScorer):
    def score_response(self, inputs: Dict[str, Any], picked: Any, event: learn_to_pick.PickBestEvent) -> float:
        # inputs: the inputs to the picker in Dict[str, Any] format
        # picked: the selection that was made by the policy
        # event: metadata that can be used to determine the score if needed
        
        # scoring logic goes here

        dummy_score = 1.0
        return dummy_score
```

#### Register callback functions after decision and before scoring

The picker will:

- Make a selection using the decision making policy
- Call the scorer to evaluate the decision
- Update the decision making policy with the score

Callback functions can be registered at create time and they will be called after the decision has been made and before the scorer is called.

```python
def a_function(inputs, picked, event):
    # I want to set the score here for some reason instead of defining a scorer
    print("hello world")
    event.selected.score = 5.0
    return inputs, event

picker = learn_to_pick.PickBest.create(
    callbacks_before_scoring = [a_function],
    selection_scorer=None
)
```

A full example can be found in `notebooks/prompt_variable_injection.ipynb`

### Store progress of learned policy

There is the option to store the decision making policy's progress and continue learning at a later time. This can be done by calling:

`picker.save_progress()`

which will store the learned policy in a file called `latest.vw`. It will also store it in a file with a timestamp. That way, if `save_progress()` is called more than once, multiple checkpoints will be created, but the latest one will always be in `latest.vw`

Next time the picker is loaded, the picker will look for a file called `latest.vw` and if the file exists it will be loaded and the learning will continue from there.

By default the model checkpoints will be stored in the current directory but the save/load location can be specified at creation time:

`picker = learn_to_pick.PickBest.create(model_save_dir=<path to dir>, [...])`

### Stop learning of learned policy

If you want the pickers learned decision making policy to stop updating you can turn it off/on:

`picker.deactivate_selection_scorer()` and `picker.activate_selection_scorer()`

### Set a different policy

There are two policies currently available:

- default policy: `VwPolicy` which learns a [Vowpal Wabbit](https://github.com/VowpalWabbit/vowpal_wabbit) [Contextual Bandit](https://github.com/VowpalWabbit/vowpal_wabbit/wiki/Contextual-Bandit-algorithms) model

- random policy: `PickBestRandomPolicy` which doesn't learn anything and just selects a value randomly. this policy can be used to compare other policies with a random baseline one.

- custom policies: a custom policy could be created and set at chain creation time

### Different exploration algorithms and options for the default learned policy

The default `VwPolicy` is initialized with some default arguments. The default exploration algorithm is [SquareCB](https://github.com/VowpalWabbit/vowpal_wabbit/wiki/Contextual-Bandit-Exploration-with-SquareCB) but other Contextual Bandit exploration algorithms can be set, and other hyper parameters can be tuned (see [here](https://vowpalwabbit.org/docs/vowpal_wabbit/python/9.6.0/command_line_args.html) for available options).

`vw_cmd = ["--cb_explore_adf", "--quiet", "--squarecb", "--interactions=::"]`

`picker = learn_to_pick.PickBest.create(vw_cmd = vw_cmd, [...])`

### Learned policy's data logs

The `VwPolicy`'s data files can be stored and examined or used to do [off policy evaluation](https://vowpalwabbit.org/docs/vowpal_wabbit/python/latest/tutorials/off_policy_evaluation.html) for hyper parameter tuning.

The way to do this is to set a log file path to `rl_logs` on chain creation:

`picker = learn_to_pick.PickBest.create(rl_logs=<path to log FILE>, [...])`

### Advanced featurization options

#### auto_embed

By default the input that is wrapped (`ToSelectFrom`, `BasedOn`) is not tampered with. This might not be sufficient featurization, so based on how complex the scenario is you can set auto-embeddings to ON

`picker = learn_to_pick.PickBest.create(featurizer=learn_to_pick.PickBestFeaturizer(auto_embed=True), [...])`

This will produce more complex embeddings and featurizations of the inputs, likely accelerating RL learning, albeit at the cost of increased runtime.

By default, [sbert.net's sentence_transformers's ](https://www.sbert.net/docs/pretrained_models.html#model-overview) `all-mpnet-base-v2` model will be used for these embeddings but you can set a different embeddings model by initializing featurizer with a different model. You could also set an entirely different embeddings encoding object, as long as it has an `encode()` function that returns a list of the encodings.

```python
from sentence_transformers import SentenceTransformer

picker = learn_to_pick.PickBest.create(
    featurizer=learn_to_pick.PickBestFeaturizer(
        auto_embed=True,
        model=SentenceTransformer("all-mpnet-base-v2")
    ),
    [...]
)
```

#### explicitly defined embeddings

Another option is to define what inputs you think should be embedded manually:

- `auto_embed = False`
- Can wrap individual variables in `learn_to_pick.Embed()` or `learn_to_pick.EmbedAndKeep()` e.g. `criteria = learn_to_pick.BasedOn(learn_to_pick.Embed("Tom"))`

#### custom featurization

Another final option is to define and set a custom featurization/embedder class that returns a valid input for the learned policy.

#### other advanced featurization options

Explictly numerical features can be provided with a colon separator:
`age = learn_to_pick.BasedOn("age:32")`

`ToSelectFrom` can be a bit more complex if the scenario demands it, instead of being a list of strings it can be:

- a list of list of strings:

    ```python
    pick = learn_to_pick.ToSelectFrom([
        ["meal 1 name", "meal 1 description"],
        ["meal 2 name", "meal 2 description"]
    ])
    ```

- a list of dictionaries:

    ```python
    pick = learn_to_pick.ToSelectFrom([
        {"name":"meal 1 name", "description" : "meal 1 description"},
        {"name":"meal 2 name", "description" : "meal 2 description"}
    ])
    ```

- a list of dictionaries containing lists:

    ```python
    pick = learn_to_pick.ToSelectFrom([
        {"name":["meal 1", "complex name"], "description" : "meal 1 description"},
        {"name":["meal 2", "complex name"], "description" : "meal 2 description"}
    ])
    ```

`BasedOn` can also take a list of strings:

```python
criteria = learn_to_pick.BasedOn(["Tom Joe", "age:32", "state of california"])
```

there is no dictionary provided since multiple variables can be supplied wrapped in `BasedOn`

Storing the data logs into a file allows the examination of what different inputs do to the data format.