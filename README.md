# Computer-Use Automation System

## Summary

This system turns a natural-language goal into a reusable automation capability
for applications that must be operated through their UI. An LLM observes a live
application, chooses actions, and completes a workflow. The successful run is
recorded as a typed, versioned JSON artifact with input parameters, outputs,
control targets, and success checks. After review, the artifact can be replayed
with new inputs without calling the model. Configurable policies restrict
navigation and actions, while an operator can pause automation, take over the
same browser session, and return control.

The implementation uses Python, Playwright, and local OCR. The end-to-end demo
covers member lookup and account opening across three synthetic banking
interfaces: server-rendered pages, web components with shadow DOM, and a
canvas-based teller workstation. The
[design report](REPORT.md) explains the architecture, trade-offs, and planned
extensions for desktop and tenant reuse.

## Key Results

The saved discovery/write runs and latest model-free lookup checks show:

| Application | Latest lookup checks | Recorded account-opening result |
|-------------|----------------------|---------------------------------|
| Responsive portal | **3/3 passed** | Discovery and replay returned the correct account |
| Web component operations | **3/3 passed** | Discovery, capability export, and replay succeeded |
| Canvas teller | **3/3 passed** | Replay created the account, but returned `Ac0000161` instead of `AC0000161` |

Each lookup check set covers two existing members and one missing member. All
nine replays returned the expected result, preserved operator/institution
context, and made no model calls.

## Project Structure

```text
computer-use-automation/
├── README.md
├── REPORT.md                       # Architecture and design decisions
├── RULES.md                        # Safety rules, guarantees, and known gaps
├── pyproject.toml                  # Python dependencies and CLI entry point
├── uv.lock                         # Locked dependency versions
├── .env.example                    # Model key and synthetic demo configuration
├── .github/workflows/ci.yml         # Lint, formatting, types, and tests
│
├── src/computeruse/
│   ├── cli.py                      # Discovery, review, and replay commands
│   ├── loop.py                     # Observe → decide → act loop
│   ├── model.py                    # LLM requests and structured decisions
│   ├── browser.py                  # Browser observation and interaction
│   ├── capability.py               # Typed capability schema and validation
│   ├── recorder.py                 # Capability construction
│   ├── recording.py                # Discovery trace and export checks
│   ├── replay.py                   # Deterministic execution and results
│   ├── policy.py                   # Route/action permissions and approval rules
│   ├── control.py                  # Session ownership and operator handoff
│   ├── models/                     # Bundled English OCR model and licenses
│   └── ...                         # Targeting, privacy, evidence, and UI helpers
│
├── scripts/
│   ├── setup.sh                    # Install dependencies and seed demo data
│   ├── start.sh                    # Start a target application
│   ├── discover.sh                 # Discover and save a workflow
│   ├── review.sh                   # Approve a reviewed capability
│   └── replay.sh                   # Replay with new inputs
│
├── evaluation/
│   ├── sites/                      # App sources ZIP, profiles, goals, contracts
│   ├── canvas/                     # Canvas teller application
│   ├── integrated.py               # Discovery/replay evaluation and fault cases
│   ├── verify_evidence.py           # Saved evidence integrity checks
│   └── ...                         # Local fixtures and evaluation helpers
│
├── evidence/
│   ├── submission/                 # Six discovery attempts and related replays
│   ├── verification/               # Latest lookup and run-consistency checks
│   └── privacy/                    # Privacy and session-handoff evidence
├── examples/                       # Example policy and synthetic capability
├── tests/                          # Unit, browser, integration, and privacy tests
└── docs/                           # Architecture diagram and diagram source
```

## Three-Stage Pipeline

| Stage | Goal | Output |
|-------|------|--------|
| 1. Discovery | Use an LLM to complete a goal through the live UI | `capability.draft.json`, discovery journal, and command log |
| 2. Review | Inspect the contract, targets, steps, and success checks; approve the flow | `capability.json` |
| 3. Replay | Execute the approved flow with new inputs and no model decisions | Typed outputs or a business/failure outcome, replay journal, and intervention log |

## Key Features

- **Goal-driven discovery** with structured model actions, step/time limits,
  policy checks, and verification against the live application.
- **Reusable JSON capabilities** with typed inputs and outputs, versioned
  schemas, explicit targets, branches, checkpoints, and review status.
- **Deterministic replay** with bounded waits, stable target resolution,
  business outcomes, and deliberate handling of uncertain writes.
- **Multiple browser surfaces** through DOM/accessibility information and
  screenshot-based OCR for interfaces without useful markup.
- **Human handoff** that pauses automation, transfers the same live session,
  records manual-action categories, and rechecks conditions on resume.
- **Safety and evidence controls** through route/action allowlists, approval
  gates for writes, secret references, sanitized logs, and structural failure
  snapshots. Their limits are documented in [RULES.md](RULES.md).

## Installation

### Prerequisites

- Python 3.12, Bash, and [uv](https://docs.astral.sh/uv/).
- Node.js 25.9 or newer. Setup downloads a checksum-verified Node.js 26.10.0
  into `.runtime/` if a compatible version is unavailable and `NODE` is unset.
- An OpenAI API key for discovery. Review, replay, and tests do not need one.
- A graphical desktop for the interactive demo. It was tested on macOS 26
  with Apple silicon. The visible Linux demo has not
  been validated. Tk provides the control window, with a browser fallback
  when Tk is unavailable.

### Setup

```bash
git clone https://github.com/MrunmayD25/computer-use-automation.git
cd computer-use-automation

# Install Python packages, Chromium, target applications, and seed data
scripts/setup.sh
```

Setup creates `.env` from `.env.example` if needed. Set the following value in
that file before discovery:

```dotenv
OPENAI_API_KEY=your_api_key_here
```

The discovery script loads `.env` automatically. Discovery
uses `gpt-6-luna` with `xhigh` reasoning effort. The direct CLI command
`computeruse discover` accepts `--model MODEL_ID` to override the model ID.
The selected model must support the existing request settings, including
function tools, image input, and `xhigh` reasoning; the CLI does not validate
compatibility. The `scripts/discover.sh` wrapper uses the default model.

The OCR model and application sources are bundled. Setup prepares the local
apps and synthetic database; no separate model or dataset download is needed.

## Reproducing the Pipeline

### Start a Target Application

The three demo surfaces use the same synthetic financial database:

| Site argument | Application | Address | Interface |
|---------------|-------------|---------|-----------|
| `responsive` | White-label responsive portal | `http://127.0.0.1:4363/` | Server-rendered pages and forms |
| `components` | Web component operations | `http://127.0.0.1:4360/` | Shadow DOM and custom controls |
| `canvas` | Canvas teller | `http://127.0.0.1:4390/` | Canvas with no accessible page structure |

In one terminal, start the responsive portal and leave it running:

```bash
scripts/start.sh responsive
```

To run the web component application, replace `responsive` with `components`
in the `start.sh`, `discover.sh`, and `replay.sh` commands; use `canvas` for
the canvas teller. Use the same site argument across all three scripts.

Run the following stages from the repository root in a second terminal.
Use a new output folder for each discovery to preserve earlier runs.

### Stage 1: Discover a Workflow

Give the agent a member-lookup goal:

```bash
scripts/discover.sh responsive lookup runs/demo
```

This uses the goal in `evaluation/sites/goals.yaml`: “As operator OP0002,
look up member NM000054 and return their membership status as one output
named membership_status.” Pass a quoted goal as the fourth argument to
customize it. Press **Start** in the control window to begin.

To supply discovery inputs individually, use the direct CLI instead of the
script. This is an alternative to the command above:

```bash
mkdir -p runs/demo
uv run --env-file .env computeruse discover --headed \
  --website http://127.0.0.1:4363/ \
  --profile evaluation/sites/profiles/white-label-responsive.yaml \
  --goal "As operator OP0002, look up member NM000038 and return their membership status as one output named membership_status." \
  --contract evaluation/sites/member-status.contract.json \
  --input member_id=NM000038 --input operator_id=OP0002 \
  --capability-id member_status \
  --save-capability runs/demo/capability.draft.json \
  --save-approved runs/demo/capability.json \
  --journal runs/demo/discovery.jsonl --log runs/demo/discovery.log
```

Keep the goal and input values consistent. If you supply any `--input`
arguments, supply every input declared by the contract. Without them, discovery
extracts inputs from the goal and asks for missing values in the control window.
The `discover.sh` script does not accept individual `--input` arguments.

For lookup replay to return `record_not_found` for a missing member, choose
**Run the checks** during discovery. Let the comparison and outcome checks
finish, and confirm the final review says **Outcome check for record not
found: branch learned** before approving the capability. If the checks are
skipped or fail to learn the branch, replay may pause for help or stop instead.
These checks run during discovery, not before each replay. Successful export
produces `runs/demo/capability.draft.json` and discovery logs.

Choose **Keep it as a draft** to follow the CLI review below. Alternatively,
**Approve for replay** saves the approved copy immediately, allowing you to
continue to Stage 3.

### Stage 2: Review and Approve

Inspect the draft's inputs, outputs, targets, steps, and checks:

```bash
uv run computeruse review --capability runs/demo/capability.draft.json
```

After reviewing it, save the approved copy:

```bash
scripts/review.sh runs/demo
```

This command records approval without another confirmation prompt and writes
`runs/demo/capability.json`. Risky actions still require separate approval
on each replay.

### Stage 3: Replay with New Inputs

The replay script accepts individual `NAME=VALUE` arguments after the run
folder. Change `member_id` to select the member and `operator_id` to select
the operator; supply both for each lookup. Quote any argument whose value
contains spaces, such as `"nickname=Emergency fund"` for account opening.

Run the capability for an existing member and then a nonexistent member:

```bash
scripts/replay.sh responsive runs/demo member_id=NM000038 operator_id=OP0002
scripts/replay.sh responsive runs/demo member_id=NM999999 operator_id=OP0002
```

Press **Start** for each replay. The first returns the member's status. The
second returns `record_not_found` if discovery learned that branch; otherwise,
replay asks for help or stops. Neither replay calls the model.

Each invocation writes `replay-N.jsonl`, `replay-N.human.jsonl`, and
`replay-N.log`. Saved logs retain structural events and placeholders; actual
task values appear in the live terminal/control window. Failure records include
the failed checks and a sanitized view of page structure.

To skip discovery and try the bundled lookup capability without an API key,
keep the responsive app running and use:

```bash
scripts/replay.sh responsive evidence/submission/responsive-lookup \
  member_id=NM000038 operator_id=OP0002
```

### Account-Opening Workflow

The `write` task adds a savings account and returns its number. With the
responsive app running, discover the workflow and select **Approve for replay**
after inspecting its final review:

```bash
scripts/discover.sh responsive write runs/account-demo
```

Alternatively, supply all five discovery inputs individually through the
direct CLI:

```bash
mkdir -p runs/account-demo
uv run --env-file .env computeruse discover --headed \
  --website http://127.0.0.1:4363/ \
  --profile evaluation/sites/profiles/white-label-responsive.yaml \
  --goal "As operator OP0002, open a new USD savings account for member NM000054 with the nickname Holiday fund and paper statements, and return the new account's number as one output named account_number." \
  --contract evaluation/sites/open-account.contract.json \
  --input member_id=NM000054 \
  --input operator_id=OP0002 \
  --input "product=USD savings" \
  --input "nickname=Holiday fund" \
  --input statement_delivery=paper \
  --no-outcome-checks \
  --capability-id open_account \
  --save-capability runs/account-demo/capability.draft.json \
  --save-approved runs/account-demo/capability.json \
  --journal runs/account-demo/discovery.jsonl \
  --log runs/account-demo/discovery.log
```

Keep the goal and input values consistent when changing them. The
`--no-outcome-checks` flag matches the write script by skipping additional
outcome-discovery runs. Press **Start**, then select **Approve for replay**
after inspecting the final review.

Replay it for another active member:

```bash
scripts/replay.sh responsive runs/account-demo \
  member_id=NM000002 operator_id=OP0002 \
  "product=USD savings" "nickname=Emergency fund" statement_delivery=paper
```

Account-opening steps require **Approve once** during discovery and replay.
For lookup, supply `member_id` and `operator_id`; writes additionally require
`product`, `nickname`, and `statement_delivery`. Replay rejects missing or
invalid inputs before opening the browser.

Use `components` or `canvas` as the site argument to try another surface,
with its server running. The write limitations are listed in Key Results.
To reset demo data before a new write experiment, preserve its evidence, stop
all site servers, and run `scripts/start.sh responsive --reset`. This reseeds
the shared database and removes accounts created by previous runs.

## Safety and Human Handoff

Profiles in `evaluation/sites/profiles/` define permitted routes, actions,
and approval rules. The model cannot override a policy denial. The control
window shows who owns the live session and supports these actions:

| Control | Behavior |
|---------|----------|
| **Stop** | Pause automation at the next safe point, then operate the same browser directly |
| **Resume / Done, continue** | Return control and recheck the current state; this does not approve a write |
| **Approve once** | Authorize one proposed action after fresh policy and target checks |
| **Terminate** | Stop the run |

Blocked actions, uncertain delivery, missing inputs, and repeated failed
attempts can request intervention. Replay does not automatically resend a
write whose delivery is uncertain.

## Verification

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run ty check
```

The tests make no live model calls. CI runs these checks on pull requests and
pushes to `main`.
