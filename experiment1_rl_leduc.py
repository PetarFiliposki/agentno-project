import json
import random
from pettingzoo.classic import leduc_holdem_v4
from llm_utils import call_llm


ACTION_ID_MAP = {
    0: "call",
    1: "raise",
    2: "fold",
    3: "check"
}


def get_state_info(env):
    obs = env.observe(env.agent_selection)
    mask = obs["action_mask"]
    legal_idx = [i for i, m in enumerate(mask) if m == 1]

    raw_state = env.unwrapped.env.get_state(
        env.unwrapped.env.get_player_id()
    )

    return legal_idx, raw_state["raw_obs"]


def summarize_history(history, window=10):
    if not history:
        return "Ова е твојата прва епизода. Немаш претходно искуство."

    recent = history[-window:]

    wins = sum(h["result"] == "win" for h in recent)
    losses = sum(h["result"] == "loss" for h in recent)

    action_outcomes = {}

    for h in recent:
        if not h["actions_taken"]:
            continue

        action = h["actions_taken"][-1]

        if action not in action_outcomes:
            action_outcomes[action] = {"win": 0, "loss": 0}

        if h["result"] == "win":
            action_outcomes[action]["win"] += 1
        elif h["result"] == "loss":
            action_outcomes[action]["loss"] += 1

    lines = [
        f"Последни {len(recent)} епизоди: "
        f"{wins} победи, {losses} порази."
    ]

    for action, result in action_outcomes.items():
        total = result["win"] + result["loss"]

        if total > 0:
            lines.append(
                f"- Последна акција '{action}': "
                f"{result['win']}/{total} победи."
            )

    return "\n".join(lines)


def build_prompt(raw_obs, legal_idx, history_summary):
    hand = raw_obs["hand"]
    public = raw_obs["public_card"] or "нема"
    my_chips = raw_obs["my_chips"]
    all_chips = raw_obs["all_chips"]

    legal_names = [
        ACTION_ID_MAP[i]
        for i in legal_idx
    ]

    return f"""
Играш Leduc Hold'em покер.

Правила:
- Шпил: J, Q, K, секоја карта во 2 бои.
- Секој играч добива една карта.
- Се открива една заедничка карта.
- Пар е најсилен, потоа K > Q > J.

ИСТОРИЈА:
{history_summary}

Користи ја историјата за да ја подобриш стратегијата.

Тековна состојба:
- Твоја карта: {hand}
- Заедничка карта: {public}
- Твои чипови: {my_chips}
- Вложувања: {all_chips}

Дозволени акции:
{legal_names}

Одговори САМО со едно од:
{legal_names}
"""


def llm_choose_action(raw_obs, legal_idx, history_summary):
    prompt = build_prompt(
        raw_obs,
        legal_idx,
        history_summary)

    text = call_llm(
        system_prompt=(
            "Ти си стратешки LLM агент "
            "кој игра Leduc Hold'em."
        ),
        user_prompt=prompt,
        max_tokens=10
    ).lower()

    for i in legal_idx:
        if ACTION_ID_MAP[i] in text:
            return i

    return random.choice(legal_idx)


def run_episode(histories, feedback):
    env = leduc_holdem_v4.env()
    env.reset()

    rewards = {
        "player_0": 0.0,
        "player_1": 0.0}

    actions = {
        "player_0": [],
        "player_1": []}

    for agent in env.agent_iter():

        obs, reward, termination, truncation, info = env.last()
        rewards[agent] += reward

        if termination or truncation:
            action = None

        else:
            legal_idx, raw_obs = get_state_info(env)

            if feedback[agent]:
                history_summary = summarize_history(
                    histories[agent]
                )
            else:
                history_summary = (
                    "Нема пристап до претходно "
                    "искуство. Одлучувај само "
                    "според моменталната состојба."
                )

            action = llm_choose_action(
                raw_obs,
                legal_idx,
                history_summary
            )

            actions[agent].append(
                ACTION_ID_MAP[action]
            )

        env.step(action)

    env.close()

    if rewards["player_0"] > rewards["player_1"]:
        result = {
            "player_0": "win",
            "player_1": "loss"}

    elif rewards["player_0"] < rewards["player_1"]:
        result = {
            "player_0": "loss",
            "player_1": "win"}

    else:
        result = {
            "player_0": "tie",
            "player_1": "tie"}

    return rewards, actions, result


def run_experiment(n_episodes=30, player_0_feedback=True, player_1_feedback=True):
    histories = {
        "player_0": [],
        "player_1": []}

    feedback = {
        "player_0": player_0_feedback,
        "player_1": player_1_feedback}

    total_rewards = {
        "player_0": 0.0,
        "player_1": 0.0}

    wins = {
        "player_0": 0,
        "player_1": 0}

    ties = 0
    win_rate = {
        "player_0": [],
        "player_1": []}

    for ep in range(n_episodes):
        rewards, actions, result = run_episode(
            histories,
            feedback
        )
        for player in ["player_0", "player_1"]:

            histories[player].append({
                "episode": ep,
                "actions_taken": actions[player],
                "result": result[player]
            })

            total_rewards[player] += rewards[player]

            if result[player] == "win":
                wins[player] += 1

        if result["player_0"] == "tie":
            ties += 1

        if (ep + 1) % 10 == 0:

            for player in ["player_0", "player_1"]:

                recent = histories[player][-10:]

                wr = sum(
                    h["result"] == "win"
                    for h in recent
                ) / len(recent)

                win_rate[player].append(wr)

        print(
            f"Епизода {ep + 1}/{n_episodes}: "
            f"{result}")

    avg_rewards = {
        player: total_rewards[player] / n_episodes
        for player in ["player_0", "player_1"]}

    return {
        "player_0_feedback": player_0_feedback,
        "player_1_feedback": player_1_feedback,
        "avg_rewards": avg_rewards,
        "wins": wins,
        "ties": ties,
        "n_episodes": n_episodes,
        "win_rate_curve": win_rate}


if __name__ == "__main__":

    print("\nLLM + feedback VS LLM без feedback")
    results_a = run_experiment(
        n_episodes=30,
        player_0_feedback=True,
        player_1_feedback=False
    )

    print("\nLLM + feedback VS LLM + feedback")
    results_b = run_experiment(
        n_episodes=30,
        player_0_feedback=True,
        player_1_feedback=True
    )

    results = {
        "feedback_vs_no_feedback": results_a,
        "feedback_vs_feedback": results_b
    }

    with open("results_experiment1.json", "w",encoding="utf-8") as f:
        json.dump(results,f,ensure_ascii=False,indent=2)

    print("\n==============================")
    print("РЕЗУЛТАТИ")
    print("==============================")

    print("\nLLM feedback VS LLM no-feedback")
    print("Player 0 просечна награда:", results_a["avg_rewards"]["player_0"])
    print("Player 1 просечна награда:", results_a["avg_rewards"]["player_1"])
    print("Player 0 победи:", results_a["wins"]["player_0"])
    print("Player 1 победи:", results_a["wins"]["player_1"])
    print("Win-rate:", results_a["win_rate_curve"])

    print("\nLLM feedback VS LLM feedback")
    print("Player 0 просечна награда:", results_b["avg_rewards"]["player_0"])
    print("Player 1 просечна награда:", results_b["avg_rewards"]["player_1"])
    print("Player 0 победи:", results_b["wins"]["player_0"])
    print("Player 1 победи:", results_b["wins"]["player_1"])
    print("Win-rate:", results_b["win_rate_curve"])