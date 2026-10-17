# Support bot: a `promptise run` project

A small support agent for an online shop, set up as a `promptise run`
project. It shows every way to run a project: as a plain program, with dev
tools, in the terminal UI, as an HTTP API and in production mode.

| File | What it is |
|---|---|
| [`promptise.toml`](promptise.toml) | The manifest: `main`, the agent factory, scripts, env, dev/prod/serve settings, prices |
| [`main.py`](main.py) | The program. Answers three sample tickets, or the one you pass |
| [`agent.py`](agent.py) | `create_agent`, the agent factory, and three local tools: `lookup_order`, `search_help_center`, `open_ticket` |

The tools use an in-memory shop. Apart from the model, nothing goes over the
network. Nothing in the code refers to `promptise run`: the dev tools, the
terminal UI and production logging are added from outside.

## Setup

```bash
pip install promptise
cd examples/run/support_bot
cp .env.example .env        # then set OPENAI_API_KEY in .env
```

## Run it

```bash
promptise run                                     # main.py: the three sample tickets
promptise run main.py "Where is my order A1002?"  # one ticket (arguments go to main.py)
python main.py                                    # the same program without promptise run
```

`python main.py` needs `OPENAI_API_KEY` in your environment or in `.env`.
`promptise run` also checks `[env] required` before it starts, and reports
every missing variable at once.

## Dev tools

```bash
promptise run --devtools      # live tool calls, tokens and cost, and a summary at the end
promptise run dev             # the same, and restart when a file changes
```

The cost estimate uses the `[pricing]` table in `promptise.toml`. Check the
numbers against your provider's price page.

## Terminal UI

```bash
promptise run --tui           # or: promptise run ui
promptise run chat            # the TUI in chat mode: talk to create_agent() directly
```

Press a digit to switch tabs (Trace, Tools, Metrics, ...), `r` to restart,
`q` to quit and `?` for help.

## HTTP API

```bash
promptise run --serve         # or: promptise run serve
```

In another terminal:

```bash
curl -s localhost:8080/v1/invoke -H 'content-type: application/json' \
  -d '{"input": "Where is my order A1002?"}'

curl -N localhost:8080/v1/stream -H 'content-type: application/json' \
  -d '{"input": "How do I return a jacket?"}'
```

## Production

```bash
promptise run check           # preflight checks only: exit 0 when ready, 2 when not
promptise run start           # JSON logs on stderr, graceful shutdown on SIGTERM
```

See [Projects & promptise run](https://docs.promptise.com/core/projects/) for
the manifest reference and every flag.
