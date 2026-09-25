"""
================================================================
ЕКСПЕРИМЕНТ 1 - Учење со поттикнување кај ЛЛМ-базирани агенти
================================================================

Играч 0: ЛЛМ-агент со "in-context reinforcement learning" - по секоја
         епизода му се дава резиме на претходните резултати (победи/порази,
         кои акции биле успешни), кое influence-ира идни одлуки преку
         контекстот (без ажурирање на тежини, туку преку prompt-меморија).
Играч 1: Случаен агент (baseline).

Мерки: просечна награда, win rate низ епизодите (крива на учење),
       споредба со baseline агент без историја (само еднократен prompt).
"""

import json
import random
from pettingzoo.classic import leduc_holdem_v4
from llm_utils import call_llm

ACTION_ID_MAP = {0: "call", 1: "raise", 2: "fold", 3: "check"}


def get_state_info(env):
    obs = env.observe(env.agent_selection)
    mask = obs["action_mask"]
    legal_idx = [i for i, m in enumerate(mask) if m == 1]
    raw_state = env.unwrapped.env.get_state(env.unwrapped.env.get_player_id())
    return legal_idx, raw_state["raw_obs"]

def summarize_history(history, window=10):
    """Гради текстуално резиме на последните `window` епизоди за feedback."""
    if not history:
        return "Ова е твојата прва епизода. Немаш претходно искуство."

    recent = history[-window:]
    wins = sum(1 for h in recent if h["result"] == "win")
    losses = sum(1 for h in recent if h["result"] == "loss")

    # grouping: кои акции почесто водат до победа
    action_outcomes = {}
    for h in recent:
        for a in h["actions_taken"]:
            action_outcomes.setdefault(a, {"win": 0, "loss": 0})
            # action_outcomes[a][h["result"] if h["result"] in ("win", "loss") else "loss"] += 0  #
        if h["actions_taken"]:
            last_a = h["actions_taken"][-1]
            action_outcomes.setdefault(last_a, {"win": 0, "loss": 0})
            if h["result"] == "win":
                action_outcomes[last_a]["win"] += 1
            elif h["result"] == "loss":
                action_outcomes[last_a]["loss"] += 1

    lines = [
        f"Резиме на последните {len(recent)} епизоди: {wins} победи, {losses} порази.",
    ]
    for a, o in action_outcomes.items():
        total = o["win"] + o["loss"]
        if total > 0:
            lines.append(f"  - Кога последната твоја акција беше '{a}': {o['win']}/{total} победи.")

    return "\n".join(lines)


def build_prompt(raw_obs, legal_idx, history_summary):
    hand = raw_obs["hand"]
    public = raw_obs["public_card"] or "нема (сè уште не е откриена)"
    my_chips = raw_obs["my_chips"]
    all_chips = raw_obs["all_chips"]
    legal_names = [ACTION_ID_MAP[i] for i in legal_idx]

    prompt = f"""Играш покер варијанта Leduc Hold'em против противник. Правила:
- Шпил од 6 карти: J, Q, K секоја во 2 бои. Секој играч добива 1 карта.
- По првиот круг на обложување се открива 1 заедничка карта.
- Победуваш ако твојата карта се совпаѓа со заедничката (пар), инаку K>Q>J.

{history_summary}

Искористи го горното резиме за да ја прилагодиш стратегијата (пр. ако 'raise'
со слаба карта често губи, биди поконзервативен; ако 'call' со силна карта
победува, продолжи со тоа).

Тековна состојба:
- Твојата карта: {hand}
- Заедничка карта: {public}
- Твоја вложена сума: {my_chips}
- Вложени суми (сите играчи): {all_chips}

Дозволени акции: {legal_names}
Одговори САМО со едно од точните имиња: {legal_names}. Без образложение."""
    return prompt


def llm_choose_action(raw_obs, legal_idx, history_summary):
    prompt = build_prompt(raw_obs, legal_idx, history_summary)
    text = call_llm(
        system_prompt="Ти си стратешки агент кој игра покер и учи од претходно искуство.",
        user_prompt=prompt,
        max_tokens=10,
    ).lower()
    for i in legal_idx:
        if ACTION_ID_MAP[i] in text:
            return i
    return random.choice(legal_idx)


def run_episode(history_summary, llm_player="player_0", render=False):
    env = leduc_holdem_v4.env(render_mode="human" if render else None)
    env.reset()

    rewards = {"player_0": 0.0, "player_1": 0.0}
    actions_taken_by_llm = []

    for agent in env.agent_iter():
        if render:
            env.render()

        obs, reward, termination, truncation, info = env.last()
        rewards[agent] += reward

        if termination or truncation:
            action = None
        else:
            legal_idx, raw_obs = get_state_info(env)
            if agent == llm_player:
                action = llm_choose_action(raw_obs, legal_idx, history_summary)
                actions_taken_by_llm.append(ACTION_ID_MAP[action])
                if render:
                    print(f">>> {agent} (LLM) избра: {ACTION_ID_MAP[action]}")
            else:
                action = random.choice(legal_idx)
                if render:
                    print(f">>> {agent} (random) избра: {ACTION_ID_MAP[action]}")

        env.step(action)

    env.close()

    result = "win" if rewards[llm_player] > rewards["player_1" if llm_player == "player_0" else "player_0"] else (
        "loss" if rewards[llm_player] < rewards["player_1" if llm_player == "player_0" else "player_0"] else "tie"
    )

    return rewards, actions_taken_by_llm, result


def run_experiment(n_episodes=50, use_feedback=True, llm_player="player_0", render=False):
    """
    use_feedback=True  -> ЛЛМ-от добива резиме од претходни епизоди (in-context RL)
    use_feedback=False -> baseline: ЛЛМ-от одлучува секој пат без историја (obicen prompting)
    """
    history = []
    total_rewards = {"player_0": 0.0, "player_1": 0.0}
    wins = {"player_0": 0, "player_1": 0}
    ties = 0
    win_rate_curve = []

    for ep in range(n_episodes):
        history_summary = summarize_history(history) if use_feedback else \
            "Одлучувај само врз основа на тековната состојба (нема пристап до претходно искуство)."

        rewards, actions_taken, result = run_episode(history_summary, llm_player=llm_player, render=render)

        history.append({"episode": ep, "actions_taken": actions_taken, "result": result})

        total_rewards["player_0"] += rewards["player_0"]
        total_rewards["player_1"] += rewards["player_1"]

        if rewards["player_0"] > rewards["player_1"]:
            wins["player_0"] += 1
        elif rewards["player_1"] > rewards["player_0"]:
            wins["player_1"] += 1
        else:
            ties += 1

        if (ep + 1) % 10 == 0:
            window = history[-10:]
            wr = sum(1 for h in window if h["result"] == "win") / len(window)
            win_rate_curve.append(wr)

        print(f"[{'feedback' if use_feedback else 'no-feedback'}] Еп.{ep+1}/{n_episodes}: {rewards} -> {result}")

    avg_rewards = {k: v / n_episodes for k, v in total_rewards.items()}

    return {
        "condition": "with_feedback" if use_feedback else "no_feedback",
        "avg_rewards": avg_rewards,
        "wins": wins,
        "ties": ties,
        "n_episodes": n_episodes,
        "win_rate_curve": win_rate_curve,
    }


if __name__ == "__main__":
    # render=True -> ја прикажува секоја рака/акција во конзолата (текстуален приказ)
    # render=False -> побрзо, без испис (препорачано кога пушташ 50 епизоди - инаку МНОГУ текст)
    print("Услов А: LLM СО in-context feedback")
    results_a = run_experiment(n_episodes=30, use_feedback=True, render=False)

    print("\nУслов Б: LLM БЕЗ feedback")
    results_b = run_experiment(n_episodes=30, use_feedback=False, render=False)

    combined = {"with_feedback": results_a, "no_feedback": results_b}
    with open("results_experiment1.json", "w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)

    print("\nСПОРЕДБА")
    print(f"Со feedback: просечна награда = {results_a['avg_rewards']['player_0']:.4f}, "
          f"победи = {results_a['wins']['player_0']}/{results_a['n_episodes']}")
    print(f"Без feedback: просечна награда = {results_b['avg_rewards']['player_0']:.4f}, "
          f"победи = {results_b['wins']['player_0']}/{results_b['n_episodes']}")
    print(f"Крива на win rate (со feedback, по прозорци од 10): {results_a['win_rate_curve']}")
    print(f"Крива на win rate (без feedback, по прозорци од 10): {results_b['win_rate_curve']}")