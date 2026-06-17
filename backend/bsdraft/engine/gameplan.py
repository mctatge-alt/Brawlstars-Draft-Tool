"""Post-draft game plan.

Heuristic strategic advice for how to *play* the drafted comp: a win condition, per-brawler
roles, how to handle each enemy threat, mode-specific do's/don'ts, and how to compensate for
composition weaknesses. Rule-based (the match data is draft→outcome only, not positional), but
grounded in standard Brawl Stars mode/role strategy.
"""
from __future__ import annotations

from typing import Dict, List

from bsdraft.engine.scoring import _class_of, _name_map
from bsdraft.engine.state import DraftState

MODE_PLAN: Dict[str, dict] = {
    "Gem Grab": {
        "objective": "Hold 10 gems through the 15-second countdown.",
        "tips": [
            "Control the center mine and assign a durable gem carrier to protect.",
            "When you hit 10, back off and play defensively for the timer.",
            "Bait out enemy Supers before committing to a fight.",
        ],
        "avoid": [
            "Dying with gems — especially near the countdown.",
            "Letting one player hoard all the gems out front.",
        ],
    },
    "Brawl Ball": {
        "objective": "Score 2 goals (or lead when time runs out).",
        "tips": [
            "Win the ball with your tank/assassin and use walls to open lanes.",
            "Save Supers for scoring or last-ditch defense.",
            "Keep one player back to defend your net.",
        ],
        "avoid": [
            "Throwing the ball blindly into a crowd.",
            "Everyone pushing up and leaving an open net.",
        ],
    },
    "Knockout": {
        "objective": "Win 2 of 3 rounds — no respawns, so every life is precious.",
        "tips": [
            "Play for picks: catch enemies out of position and win the numbers game.",
            "Use bushes and cover; poke from range before committing.",
            "Rotate together — don't get isolated.",
        ],
        "avoid": [
            "Face-checking bushes alone.",
            "Over-aggression — one death cripples the whole round.",
        ],
    },
    "Heist": {
        "objective": "Break the enemy safe before they break yours.",
        "tips": [
            "Bring burst to the safe when the enemy Supers are down.",
            "Split pressure to force them to choose what to defend.",
            "Track enemy Supers and push on the gap.",
        ],
        "avoid": [
            "Leaving your safe undefended.",
            "Feeding Supers by trading badly.",
        ],
    },
    "Hot Zone": {
        "objective": "Control the zone(s) to fill your meter faster.",
        "tips": [
            "Use area-control brawlers to hold the zone and deny theirs.",
            "Rotate to contest, but only fight where you can win.",
            "Stack the zone when ahead to close it out.",
        ],
        "avoid": [
            "Chasing kills off the zone.",
            "Clumping into throwers / AoE.",
        ],
    },
    "Bounty": {
        "objective": "Lead on stars when time runs out (7-star cap).",
        "tips": [
            "Poke from range and pick off high-bounty targets.",
            "When ahead, play safe and protect your star lead.",
            "Group up early to avoid giving first blood.",
        ],
        "avoid": [
            "Feeding stars by over-extending.",
            "Fighting at a star disadvantage.",
        ],
    },
}

ROLE_BY_CLASS = {
    "Tank": "Frontline — soak damage, create space, dive their backline.",
    "Assassin": "Flanker — catch isolated targets and delete their carry.",
    "Marksman": "Backline — hold range and deal damage from safety.",
    "Controller": "Zoner — lock down chokes and control space.",
    "Artillery": "Poke — chip from range and deny areas with throws.",
    "Support": "Enabler — peel for and pocket your carry.",
    "Damage Dealer": "Damage — trade from cover and burst priority targets.",
    "Unclassified": "Flex — play to this brawler's strengths.",
}

THREAT_BY_CLASS = {
    "Tank": "kite {name} and keep distance; focus it when its Super is down.",
    "Assassin": "group up and watch bushes for {name}; don't get caught alone or low.",
    "Marksman": "use cover against {name}; don't walk open lanes — dive it with frontline.",
    "Controller": "respect {name}'s zoning/CC and don't clump up; flank when you can.",
    "Artillery": "close the distance on {name} or break its walls — don't sit in its throw zone.",
    "Support": "pressure or dive {name}, or focus the brawler it's pocketing.",
    "Damage Dealer": "respect {name}'s burst; trade from cover, don't face-tank it.",
}

_TONE = {
    "Aggressive": "Be proactive",
    "Control / poke": "Be patient",
    "Balanced": "Stay flexible",
}
_MODE_VERB = {
    "Gem Grab": "control the mine and protect your carry",
    "Brawl Ball": "win the ball and convert with your frontline",
    "Knockout": "play for picks and win the numbers game",
    "Heist": "burst the safe when their Supers are down",
    "Hot Zone": "hold the zone and out-rotate them",
    "Bounty": "out-poke them and protect your star lead",
}


def _archetype(classes: List[str]):
    aggro = sum(c in ("Tank", "Assassin") for c in classes)
    rangey = sum(c in ("Marksman", "Controller", "Artillery") for c in classes)
    if aggro >= 2 and rangey == 0:
        return "Aggressive", "Force fights and close distance — snowball early picks and pressure the objective."
    if rangey >= 2 and aggro == 0:
        return "Control / poke", "Play your range advantage: poke, zone, and punish over-extension. Avoid melee brawls."
    return "Balanced", "Let your frontline engage and your backline follow up — win the range war, then commit."


def game_plan(state: DraftState) -> dict:
    names = _name_map()
    our_cls = [_class_of(b) for b in state.our_team]
    archetype, playstyle = _archetype(our_cls)
    plan = MODE_PLAN.get(state.mode, {"objective": "", "tips": [], "avoid": []})

    roles = [
        {"name": names.get(b, str(b)), "cls": _class_of(b),
         "role": ROLE_BY_CLASS.get(_class_of(b), ROLE_BY_CLASS["Unclassified"])}
        for b in state.our_team
    ]
    threats = []
    for e in state.their_team:
        cls = _class_of(e)
        tip = THREAT_BY_CLASS.get(cls)
        if tip:
            threats.append({"name": names.get(e, str(e)), "cls": cls,
                            "tip": tip.format(name=names.get(e, str(e)))})

    compensate = []
    if our_cls and not any(c in ("Tank", "Assassin") for c in our_cls):
        compensate.append("No frontline — you can't contest space head-on; poke and kite, don't get dived.")
    if our_cls and not any(c in ("Marksman", "Controller", "Artillery") for c in our_cls):
        compensate.append("No long range — close distance fast and avoid poke wars you'll lose.")
    if sum(1 for b in state.their_team if _class_of(b) == "Tank") >= 2 and "Marksman" not in our_cls:
        compensate.append("Enemy is tank-heavy — kite relentlessly, chip them down, never get cornered.")

    tone = _TONE.get(archetype, "Stay flexible")
    verb = _MODE_VERB.get(state.mode, "play to your comp's strengths")
    win_condition = f"{tone}: {verb}."

    return {
        "objective": plan["objective"],
        "win_condition": win_condition,
        "archetype": archetype,
        "playstyle": playstyle,
        "roles": roles,
        "threats": threats,
        "tips": plan["tips"],
        "avoid": plan["avoid"],
        "compensate": compensate,
    }
