"""
agent.py - YOUR poker bot. This one file is a complete web server.

Run it:
    python agent.py                  # listens on port 5008
    PORT=5010 python agent.py        # or pick a port

Play a match against a baseline (three windows):
    python agent.py                                 # your bot on port 5008
    python baseline-bots/baseline_random.py         # opponent on port 5009
    python judge_engine.py                          # judge: both halves (uses those two ports)

Check that the tournament can run it:
    python -m pytest -q              # tests in test_agent.py
    ./tools/check_build.sh           # builds and smoke-tests it like the tournament

TODO:
    1. Assign your agent a unique PARTICIPANT and AGENT_NAME below.
    2. Implement the three functions between "YOUR CODE HERE" and "END OF YOUR CODE".

Everything under "SERVER PLUMBING" talks to the judge. You normally never
need to change it, and if you break it the judge can't talk to your bot.
"""

import os
import traceback

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from evaluator import card_rank, card_suit, cards_to_str, omaha_score, warmup  # handy helpers, see evaluator.py
from game import Action, AgentConfig, HandResult, Observation, Street

# Build omaha_score's lookup table now, while the server starts, instead of
# during your first decision (where it would eat into your time bank).
# Do any other slow setup (loading a model, ...) here too: the server must
# answer within 15 seconds of starting.
warmup()

# rename as your team/bot's name
PARTICIPANT = "SET_YOUR_REGISTERED_TEAM_NAME"
AGENT_NAME = "FortressML_Value_v2"


# ===== YOUR CODE HERE =====
# Fortress ML v2: fast offline-trained gradient-boosted preflop equity model,
# inexpensive postflop equity sampling, and Bayesian opponent adaptation.
# Training and its scikit-learn dependency are OFFLINE ONLY; inference is
# dependency free. The tournament's 8-second time bank is never used to train.
import random
from preflop_model import preflop_equity, warmup_model

warmup_model()

# A sample requires two 100-lookup Omaha evaluations. Keep counts small.
FLOP_TRIALS = 16
TURN_TRIALS = 20
RIVER_TRIALS = 26


def new_half(config: AgentConfig, memory: dict) -> None:
    memory.update({
        'hand_index': -1,
        'observed_actions': 0,
        'opp_raises': 0,
        'opp_checks_calls': 0,
        'opp_calls_to_our_bets': 0,
        'opp_folds_to_our_bets': 0,
        'hand_raise_pending': False,
        'equity_key': None,
        'equity_val': .5,
        # Reproducibility without sharing state between halves.
        'rng': random.Random(910731),
    })


def _opp_aggression(memory):
    # Beta posterior, with a non-extreme prior and meaningful shrinkage.
    a = memory['opp_raises']
    c = memory['opp_checks_calls']
    return (4.0 + a) / (12.0 + a + c)


def _opp_fold_frequency(memory):
    # Bayesian prior of ~22% and an extra stability requirement for adaptation.
    f = memory['opp_folds_to_our_bets']
    c = memory['opp_calls_to_our_bets']
    return (2.0 + f) / (9.0 + f + c)


def _observe(obs, memory):
    if memory['hand_index'] != obs.hand_index:
        memory['hand_index'] = obs.hand_index
        memory['observed_actions'] = 0
        memory['hand_raise_pending'] = False
        memory['equity_key'] = None
    hist = obs.action_history
    for i in range(memory['observed_actions'], len(hist)):
        h = hist[i]
        if h.is_me:
            continue
        if h.action in (Action.RAISE_HALF, Action.RAISE_POT):
            memory['opp_raises'] += 1
        elif h.action == Action.CHECK_CALL:
            memory['opp_checks_calls'] += 1
            # Only count calls that actually faced our previous raise/bet.
            if i > 0 and hist[i-1].is_me and hist[i-1].action in (
                Action.RAISE_HALF, Action.RAISE_POT
            ) and h.amount > 0:
                memory['opp_calls_to_our_bets'] += 1
    memory['observed_actions'] = len(hist)


def _equity(obs, memory):
    # Cache on the same board: no need to re-simulate for an opponent's re-raise.
    key = (obs.hand_index, obs.board)
    if key == memory['equity_key']:
        return memory['equity_val']

    trials = (FLOP_TRIALS if obs.street == Street.FLOP else
              TURN_TRIALS if obs.street == Street.TURN else RIVER_TRIALS)
    known = set(obs.hole_cards) | set(obs.board)
    deck = [c for c in range(36) if c not in known]
    missing = 5 - len(obs.board)
    rng = memory['rng']
    wins = 0.0
    mine_on_river = omaha_score(obs.hole_cards, obs.board) if missing == 0 else None
    for _ in range(trials):
        sampled = rng.sample(deck, missing + 5)
        opp = sampled[:5]
        board = obs.board if not missing else obs.board + tuple(sampled[5:])
        mine = mine_on_river if not missing else omaha_score(obs.hole_cards, board)
        other = omaha_score(opp, board)
        wins += (mine > other) + .5 * (mine == other)

    equity = (wins + 1.0) / (trials + 2.0)  # smoothing
    memory['equity_key'] = key
    memory['equity_val'] = equity
    return equity


def _preflop(obs, memory):
    e = preflop_equity(obs.hole_cards)
    legal = obs.legal_actions
    aggr = _opp_aggression(memory)
    if obs.to_call:
        # Cheap calls are very valuable in heads-up poker; tight folding can
        # otherwise lose blinds repeatedly. Require higher equity to risk more.
        cost_fraction = obs.to_call / max(1, obs.pot + obs.to_call)
        cushion = 0.035 if obs.to_call <= 4 else .075
        if obs.to_call >= 18:
            cushion += .09
        # If raises are common they convey less information about hand strength.
        defensive_floor = .32 + max(-.055, min(.055, .34 - aggr))
        if e < max(cost_fraction + cushion, defensive_floor):
            return Action.FOLD if Action.FOLD in legal else Action.CHECK_CALL
        if obs.to_call >= 22 and e < (.62 if aggr >= .6 else .70):
            return Action.FOLD if Action.FOLD in legal else Action.CHECK_CALL
        if obs.to_call >= 10 and e < (.52 if aggr >= .55 else .59):
            return Action.FOLD if Action.FOLD in legal else Action.CHECK_CALL

    # Prefer cheap half-pot value opens; pot raises require very rare premiums.
    if obs.raises_this_street == 0 and obs.to_call <= 2:
        if e >= .68 and Action.RAISE_POT in legal:
            return Action.RAISE_POT
        if e >= (.585 if obs.is_button else .61) and Action.RAISE_HALF in legal:
            return Action.RAISE_HALF
    if obs.raises_this_street == 1 and e >= .705 and obs.to_call <= 6:
        if Action.RAISE_HALF in legal:
            return Action.RAISE_HALF
    return Action.CHECK_CALL


def _postflop(obs, memory):
    e = _equity(obs, memory)
    legal = obs.legal_actions
    pot = max(1, obs.pot)
    cost = obs.to_call
    aggr = _opp_aggression(memory)
    raises_now = sum(1 for a in obs.action_history if not a.is_me and
                     a.street == obs.street and
                     a.action in (Action.RAISE_HALF, Action.RAISE_POT))

    if cost:
        pot_odds = cost / (pot + cost)
        # Slightly adjust a random-hand equity for observed betting behavior.
        # This is NOT a solved conditional range: demanding more confidence
        # in high-cost pots is the deliberate, conservative risk policy.
        risk = .035 if cost <= 4 else (.065 if cost <= 14 else .09)
        if cost > 30:
            risk += .05
        # Lower risk premium against a habitual raiser; higher when very tight.
        risk += max(-.05, min(.055, (.38 - aggr) * .12))
        if raises_now >= 2:
            risk += .02
        if e < pot_odds + risk:
            return Action.FOLD if Action.FOLD in legal else Action.CHECK_CALL
        if cost >= 24 and e < (.57 if aggr >= .65 else .69):
            return Action.FOLD if Action.FOLD in legal else Action.CHECK_CALL
        if cost >= 12 and e < (.49 if aggr >= .65 else .56):
            return Action.FOLD if Action.FOLD in legal else Action.CHECK_CALL
        # Maximal re-raise only when we are an overwhelming favorite.
        if (e >= .91 and cost <= 18 and Action.RAISE_POT in legal
                and obs.my_stack >= 5):
            return Action.RAISE_POT
        if e >= .83 and cost <= 10 and Action.RAISE_HALF in legal:
            return Action.RAISE_HALF
        return Action.CHECK_CALL

    # Bet for value when likely to be called by worse hands. Unlike blindly
    # raising with 'a flush', this uses actual Omaha equity vs random holdings.
    value = {Street.FLOP: .70, Street.TURN: .735, Street.RIVER: .765}[obs.street]
    if e >= value and Action.RAISE_HALF in legal:
        proposed = obs.raise_to.get(Action.RAISE_HALF, obs.my_bet) - obs.my_bet
        if (proposed <= 32 or e >= .87):
            if e >= .89 and Action.RAISE_POT in legal:
                pot_raise = obs.raise_to.get(Action.RAISE_POT, obs.my_bet) - obs.my_bet
                if pot_raise <= 38 or e >= .955:
                    return Action.RAISE_POT
            return Action.RAISE_HALF

    # Very rare bluff, supported by a posterior from 12+ observed responses.
    fold_events = (memory['opp_folds_to_our_bets'] +
                   memory['opp_calls_to_our_bets'])
    if (obs.street == Street.RIVER and pot <= 16
            and obs.raises_this_street == 0 and .16 <= e <= .36
            and fold_events >= 12 and _opp_fold_frequency(memory) > .64
            and Action.RAISE_HALF in legal and memory['rng'].random() < .003):
        return Action.RAISE_HALF
    return Action.CHECK_CALL


def choose_action(obs: Observation, memory: dict) -> Action:
    safe = (Action.CHECK_CALL if obs.to_call == 0 else Action.FOLD)
    if safe not in obs.legal_actions:
        safe = obs.legal_actions[0]
    try:
        _observe(obs, memory)
        choice = (_preflop(obs, memory) if obs.street == Street.PREFLOP
                  else _postflop(obs, memory))
        if choice not in obs.legal_actions:
            choice = safe
        memory['hand_raise_pending'] = (choice in (Action.RAISE_HALF, Action.RAISE_POT))
        return choice
    except Exception:
        # Tournament faults for illegal actions/exceptions are much worse than
        # folding a marginal hand. Remain legal if unexpected inputs occur.
        memory['hand_raise_pending'] = False
        return safe


def on_hand_end(result: HandResult, memory: dict) -> None:
    try:
        if (result.hand_index == memory.get('hand_index')
                and memory.get('hand_raise_pending')
                and not result.went_to_showdown and result.winner == 'me'):
            memory['opp_folds_to_our_bets'] += 1
        memory['hand_raise_pending'] = False
    except Exception:
        pass
# ===== END OF YOUR CODE =====



# ===== SERVER PLUMBING (you normally don't need to edit below) =====
# The judge calls these routes:
#   GET  /           -> who you are (also used as the health check)
#   POST /new-half   -> start a fresh half: memory is wiped
#   POST /act        -> the judge sends any finished hands plus an Observation;
#                       on_hand_end() runs for each hand, then you answer an action
#   POST /hand-end   -> the judge sends the last finished hands of a half
#
# Why "async def"? The routes run on one thread and call your functions
# directly, so your functions never run twice at once. Your own functions
# stay normal "def" (don't make them async).
#
# Keep your per-half state in `memory`. In the tournament every half runs in
# a fresh server process, so everything starts over. When you test locally,
# though, one process can play both halves and only `memory` is wiped, NOT
# other global variables. Using `memory` makes your bot behave the same way
# in both places.

app = FastAPI()
MEMORY: dict = {}                                   # your memory for this half
STATUS = {"started": False}                         # has /new-half been called?
TOKEN = os.environ.get("AGENT_TOKEN") or None       # set by the tournament


async def _read_request(request: Request, path: str) -> tuple[dict, JSONResponse | None]:
    """Check the token (and that the half has started) BEFORE reading the body.

    Checking first means a request without the right token costs your bot
    almost nothing, so nobody else can slow it down by sending junk.
    Returns (body, None) when the request is fine, or ({}, error response).
    """
    if TOKEN and request.headers.get("x-agent-token") != TOKEN:
        return {}, JSONResponse({"error": "bad or missing X-Agent-Token"}, status_code=401)
    if not STATUS["started"] and path != "/new-half":
        return {}, JSONResponse({"error": "call /new-half first"}, status_code=409)
    try:
        body = await request.json()
    except ValueError:
        body = None
    if not isinstance(body, dict):
        return {}, JSONResponse({"error": "the request body must be a JSON object"}, status_code=400)
    return body, None


def _error_info(exc: Exception) -> dict:
    """Describe an exception from your code (call this inside `except`)."""
    return {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}


def _crash(exc: Exception, hand_end_errors: list | None = None) -> JSONResponse:
    """Report an exception from your code to the judge (with traceback)."""
    return JSONResponse({**_error_info(exc), "hand_end_errors": hand_end_errors or []},
                        status_code=500)


def _deliver_results(raw_results) -> list:
    """Call on_hand_end() for each finished hand the judge sent.
    Returns a list of error reports (one per hand that crashed)."""
    errors = []
    for raw in raw_results or []:
        try:
            on_hand_end(HandResult.from_dict(raw), MEMORY)
        except Exception as exc:
            errors.append(_error_info(exc))
    return errors


@app.get("/")
async def info():
    """Health check: tells the judge who this bot is."""
    return {"participant": PARTICIPANT, "agent_name": AGENT_NAME}


@app.post("/new-half")
async def route_new_half(request: Request):
    """Wipe memory and call your new_half(). The body is {"config": {...}}."""
    body, refused = await _read_request(request, "/new-half")
    if refused:
        return refused
    MEMORY.clear()
    STATUS["started"] = True
    try:
        new_half(AgentConfig.from_dict(body.get("config", {})), MEMORY)
    except Exception as exc:
        return _crash(exc)
    return {"ok": True}


@app.post("/act")
async def route_act(request: Request):
    """Deliver finished hands, then turn the JSON into an Observation, call
    choose_action(), and send the answer back."""
    body, refused = await _read_request(request, "/act")
    if refused:
        return refused
    hand_end_errors = _deliver_results(body.get("hand_results"))
    try:
        obs = Observation.from_dict(body["observation"])
        action = choose_action(obs, MEMORY)
    except Exception as exc:
        return _crash(exc, hand_end_errors)
    # Send the action's name; anything strange is passed on for the judge to reject.
    if isinstance(action, Action):
        answer = action.name
    elif action is None or isinstance(action, (str, int, float, bool)):
        answer = action
    else:
        answer = repr(action)
    return {"action": answer, "hand_end_errors": hand_end_errors}


@app.post("/hand-end")
async def route_hand_end(request: Request):
    """Call on_hand_end() for the last finished hands of a half."""
    body, refused = await _read_request(request, "/hand-end")
    if refused:
        return refused
    return {"ok": True, "hand_end_errors": _deliver_results(body.get("results"))}


if __name__ == "__main__":
    # The tournament sets PORT as an environment variable, so each bot is compliant to our evaluation platform
    # Locally it defaults to 5008 for your own testing convenience
    port = int(os.environ.get("PORT", "5008"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info", access_log=False)