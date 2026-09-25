"""
ЕКСПЕРИМЕНТ 2 - Координација, преговори и дебата кај повеќеагентски ЛЛМ-системи
================================================================================
Задача: "Поделба на проектни ресурси" - два ЛЛМ-агенти (Агент А и Агент Б)
преговараат како да поделат 3 ресурси (буџет, тим, временска рамка), секој
со приватни (скриени за другиот) тежини на важност кои сумираат до 1.0.

Услов 1 (со дебата): пред секоја понуда, агентите разменуваат кратка критика/
         аргумент за претходната понуда (дебатен круг), па потоа ревидираат.
Услов 2 (без дебата, baseline): агентите директно разменуваат понуди,
         без дебатен круг на критика/аргументација.

Мерки: стапка на постигнат договор, просечна корист (utility) по агент,
       Nash product (мерка за фер-распределба), просечен број рунди.
"""

import json
import re
import random
from llm_utils import call_llm

ISSUES = ["буџет", "тим", "временска рамка"]
MAX_ROUNDS = 5
RESERVATION_UTILITY = 0.0


def random_weights():
    """Генерира случаен вектор на приватни тежини кои сумираат до 1.0."""
    raw = [random.random() for _ in ISSUES]
    total = sum(raw)
    return {issue: round(r / total, 2) for issue, r in zip(ISSUES, raw)}


def utility(allocation_for_agent, weights):
    """Kолку сум задоволен/среќен -> % (0-100) добиени од агентот."""
    return sum(weights[issue] * allocation_for_agent[issue] / 100 for issue in ISSUES)


def parse_offer(text):
    """Очекува понуда во формат JSON: {"буџет": 60, "тим": 40, "временска рамка": 70}"""
    match = re.search(r"\{[^{}]*\}", text)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
        if all(issue in data for issue in ISSUES):
            return {issue: max(0, min(100, float(data[issue]))) for issue in ISSUES}
    except (json.JSONDecodeError, ValueError, TypeError):
        return None
    return None


def check_agreement(text):
    return "се согласувам" in text.lower() or "прифаќам" in text.lower() or "agree" in text.lower()


def build_offer_prompt(agent_name, weights, last_offer_from_other, debate_note, round_num):
    other_offer_str = (
        f"Претходна понуда од другиот агент (% за ТЕБЕ по секој ресурс): {last_offer_from_other}"
        if last_offer_from_other else "Ова е првата рунда, нема претходна понуда."
    )
    debate_str = f"\nТвојата белешка од дебатата пред оваа понуда: {debate_note}" if debate_note else ""

    return f"""Ти си {agent_name} во преговори за поделба на 3 проектни ресурси меѓу
тебе и друг агент: {ISSUES}. За секој ресурс се одредува процент (0-100) кој
припаѓа НА ТЕБЕ (остатокот му припаѓа на другиот агент).

Твои ПРИВАТНИ тежини на важност (не ги знае другиот агент): {weights}
Твојата вкупна корист = збир(тежина_i × твој_процент_i / 100), максимум 1.0.

{other_offer_str}{debate_str}

Рунда {round_num}/{MAX_ROUNDS}.

Ако сакаш да ја ПРИФАТИШ претходната понуда, одговори со зборот "прифаќам".
Инаку, дај НОВА понуда како JSON со процент за ТЕБЕ по ресурс, пр:
{{"буџет": 60, "тим": 40, "временска рамка": 70}}
Одговори САМО со зборот "прифаќам" ИЛИ со JSON, без друго објаснување."""


def build_debate_prompt(agent_name, weights, opponent_offer):
    return f"""Ти си {agent_name}. Другиот агент во преговорите ти предложи да добиеш
следниве проценти по ресурс: {opponent_offer} (ресурси: {ISSUES}).
Твоите приватни тежини на важност се: {weights}.

Во 1-2 реченици, критикувај ја понудата од своја перспектива и наведи
кој ресурс сакаш да добиеш поголем удел и зошто (ова ќе го искористиш
за да ја обликуваш следната своја понуда). Кратко, без JSON."""


def run_negotiation(use_debate, verbose=False):
    weights_a = random_weights()
    weights_b = random_weights()

    agents = ["Агент А", "Агент Б"]
    weights = {"Агент А": weights_a, "Агент Б": weights_b}

    last_offer = None
    offering_agent_idx = 0

    for round_num in range(1, MAX_ROUNDS + 1):
        current_agent = agents[offering_agent_idx % 2]
        other_agent = agents[(offering_agent_idx + 1) % 2]

        debate_note = None
        if use_debate and last_offer is not None:
            debate_prompt = build_debate_prompt(current_agent, weights[current_agent], last_offer)
            debate_note = call_llm(
                system_prompt="Ти си стратешки преговарач кој гради аргументи пред да понуди.",
                user_prompt=debate_prompt,
                max_tokens=150,
            )
            if verbose:
                print(f"  [дебата - {current_agent}]: {debate_note}")

        offer_prompt = build_offer_prompt(current_agent, weights[current_agent], last_offer, debate_note, round_num)
        response = call_llm(
            system_prompt="Ти си стратешки преговарач за поделба на ресурси.",
            user_prompt=offer_prompt,
            max_tokens=100,
        )

        if verbose:
            print(f"  [понуда - {current_agent}]: {response}")

        if check_agreement(response) and last_offer is not None:
            # тековниот агент ја прифаќа понудата (last_offer е % за него од претходна рунда)
            allocation_current = last_offer
            allocation_other = {issue: 100 - allocation_current[issue] for issue in ISSUES}

            if current_agent == "Агент А":
                alloc_a, alloc_b = allocation_current, allocation_other
            else:
                alloc_b, alloc_a = allocation_current, allocation_other

            u_a = utility(alloc_a, weights_a)
            u_b = utility(alloc_b, weights_b)

            return {
                "agreement": True,
                "rounds": round_num,
                "utility_a": u_a,
                "utility_b": u_b,
                "nash_product": u_a * u_b,
                "allocation_a": alloc_a, # финалната, стварна поделба на ресурсите откако е постигнат договор
                "allocation_b": alloc_b,
            }

        parsed = parse_offer(response)
        if parsed is None:
            # LLM не одговори во валиден формат - третираме како неуспешна рунда, продолжуваме
            parsed = {issue: 50 for issue in ISSUES}

        last_offer = parsed  # % за СЛЕДНИОТ агент (примателот) - според дефиницијата подолу
        last_offer = {issue: 100 - parsed[issue] for issue in ISSUES}

        offering_agent_idx += 1

    # нема договор по MAX_ROUNDS
    return {
        "agreement": False,
        "rounds": MAX_ROUNDS,
        "utility_a": RESERVATION_UTILITY,
        "utility_b": RESERVATION_UTILITY,
        "nash_product": 0.0,
        "allocation_a": None,
        "allocation_b": None,
    }


def run_experiment(n_trials=30, use_debate=True, verbose=False):
    results = []
    for t in range(n_trials):
        res = run_negotiation(use_debate=use_debate, verbose=verbose)
        results.append(res)
        print(f"[{'debate' if use_debate else 'no-debate'}] Обид {t+1}/{n_trials}: "
              f"agreement={res['agreement']}, rounds={res['rounds']}, "
              f"U_A={res['utility_a']:.2f}, U_B={res['utility_b']:.2f}")

    agreements = [r for r in results if r["agreement"]]
    agreement_rate = len(agreements) / n_trials
    avg_rounds = sum(r["rounds"] for r in results) / n_trials
    avg_utility_a = sum(r["utility_a"] for r in results) / n_trials
    avg_utility_b = sum(r["utility_b"] for r in results) / n_trials
    avg_nash = sum(r["nash_product"] for r in results) / n_trials

    return {
        "condition": "with_debate" if use_debate else "no_debate",
        "n_trials": n_trials,
        "agreement_rate": agreement_rate,
        "avg_rounds": avg_rounds,
        "avg_utility_a": avg_utility_a,
        "avg_utility_b": avg_utility_b,
        "avg_nash_product": avg_nash,
        "raw_results": results,
    }


if __name__ == "__main__":
    print("=== Услов А: СО дебатен круг пред секоја понуда ===")
    results_debate = run_experiment(n_trials=2, use_debate=True, verbose=True)

    print("\n=== Услов Б: БЕЗ дебатен круг (директно преговарање) ===")
    results_no_debate = run_experiment(n_trials=2, use_debate=False, verbose=True)

    combined = {"with_debate": results_debate, "no_debate": results_no_debate}
    with open("results_experiment2.json", "w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)

    print("\n===== СПОРЕДБА =====")
    print(f"Со дебата: agreement_rate={results_debate['agreement_rate']:.2f}, "
          f"avg_nash={results_debate['avg_nash_product']:.3f}, avg_rounds={results_debate['avg_rounds']:.2f}")
    print(f"Без дебата: agreement_rate={results_no_debate['agreement_rate']:.2f}, "
          f"avg_nash={results_no_debate['avg_nash_product']:.3f}, avg_rounds={results_no_debate['avg_rounds']:.2f}")