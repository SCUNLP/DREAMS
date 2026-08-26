# DREAMS

Code for **Towards Structured Context Modeling for Conversational Recommender Systems via Dual Monte Carlo Tree Search**.

All commands below should be run from the repository root.

## 1. Install

```bash
git clone https://github.com/SCUNLP/DREAMS.git
cd DREAMS

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## 2. Configure the API

```bash
export OPENAI_API_KEY="your-api-key"
```

Optional settings for an OpenAI-compatible endpoint:

```bash
export OPENAI_BASE_URL="https://your-endpoint.example/v1"
export OPENAI_CHAT_MODEL="gpt-4o-mini"
export OPENAI_EMBEDDING_MODEL="text-embedding-ada-002"
```

## 3. Build item embeddings

Build the cache for the dataset you want to run:

```bash
# ReDial
python script/cache_item.py --dataset redial --batch_size 1000

# OpenDialKG
python script/cache_item.py --dataset opendialkg --batch_size 1000
```

Embeddings are saved under `save/embed/item/<dataset>/`.

## 4. Run the web demo

```bash
python code/web_demo.py \
  --dataset redial \
  --mcts_iterations 6 \
  --simulation_workers 3
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000) if the browser does not open automatically.

For OpenDialKG:

```bash
python code/web_demo.py --dataset opendialkg
```

## 5. Run command-line interaction

ReDial:

```bash
python code/interactive_dreams.py \
  --dataset redial_eval \
  --kg_dataset redial \
  --crs_model mcts_dual \
  --mcts_iterations 3
```

OpenDialKG:

```bash
python code/interactive_dreams.py \
  --dataset opendialkg_eval \
  --kg_dataset opendialkg \
  --crs_model mcts_dual \
  --mcts_iterations 3
```

Enter `/prefs` to show the current preference state or `/exit` to stop.

## 6. Run user-simulator evaluation

```bash
bash code/dual_chat_eval.sh \
  --dataset redial_eval \
  --kg_dataset redial \
  --crs_model mcts_dual \
  --mcts_iterations 3
```

Use the matching pair `--dataset opendialkg_eval --kg_dataset opendialkg` for OpenDialKG.

## Optional arguments

```text
--mcts_iterations N       MCTS iterations from turn three onward
--simulation_workers N    concurrent rollout workers; default: 1
--seed N                  random seed; default: 100
--trace                   generate the standalone MCTS trace page
--no_trace_open           do not open the trace page automatically
```

The first two conversation turns ask preference questions directly and do not run simulation or backpropagation.

## Tests

```bash
python -m unittest discover -s tests
python -m compileall -q code script data
```
