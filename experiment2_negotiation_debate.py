import json
import re
import random
from llm_utils import call_llm

ISSUES = ["буџет", "тим", "временска рамка"]
MAX_ROUNDS = 5

def random_weights():
    r = [random.random() for _ in ISSUES]
    s = sum(r)
    return {i: round(x / s, 2) for i, x in zip(ISSUES, r)}

def utility(a, w):
    return sum(w[i] * a[i] / 100 for i in ISSUES)

def parse_offer(text):
    m = re.search(r"\{[^{}]*\}", text)
    if not m:
        return None
    try:
        d = json.loads(m.group())
        if all(i in d for i in ISSUES):
            return {i: max(0, min(100, float(d[i]))) for i in ISSUES}
    except:
        pass
    return None

def accepted(text):
    return "прифаќам" in text.lower() or "agree" in text.lower()

def offer_prompt(agent, weights, received, context, rnd):
    previous = f"Понуда за тебе: {received}" if received else "Нема претходна понуда."
    ctx = f"\n{context}" if context else ""

    return f"""
Ти си {agent}. Преговараш со друг LLM агент за {ISSUES}.
Твои приватни тежини: {weights}
{previous}{ctx}
Рунда {rnd}/{MAX_ROUNDS}.

Ако ја прифаќаш понудата напиши само "прифаќам".
Инаку даде нова понуда во JSON. Процентите се за тебе.
Пример: {{"буџет":60,"тим":40,"временска рамка":70}}
Одговори само со "прифаќам" или JSON.
"""

def argument_prompt(agent, weights, offer):
    return f"""
Ти си {agent}. Другиот агент ти понуди: {offer}.
Твои тежини: {weights}.
Во 1-2 реченици, аргументирај конкретно зошто сметаш дека оваа понуда
е неповолна за тебе и кој ресурс бараш да се промени и зошто (повикај
се на твоите приватни тежини). Ова е твој отворен аргумент кон другиот агент.
"""

def rebuttal_prompt(other_agent, other_weights, opponent_argument):
    return f"""
Ти си {other_agent}. Другиот агент ти го изнесе следниот аргумент
зошто бара поголем удел: "{opponent_argument}"

Твои приватни тежини: {other_weights}.
Одговори конкретно на тој аргумент (прифати го делумно, отфрли го,
или понуди компромис) во 1-2 реченици. Ова е твој контра-аргумент
(rebuttal), не нова понуда.
"""

def negotiate(debate=True, verbose=False):
    weights = {"Агент А": random_weights(), "Агент Б": random_weights()}
    agents = ["Агент А", "Агент Б"]
    received = None

    for rnd in range(1, MAX_ROUNDS + 1):
        agent = agents[(rnd - 1) % 2]
        other = agents[rnd % 2]

        context = None

        if debate and received:
            argument = call_llm(
                "Ти си преговарач кој изнесува аргумент.",
                argument_prompt(agent, weights[agent], received),
                120
            )
            rebuttal = call_llm(
                "Ти си преговарач кој одговара на туѓ аргумент.",
                rebuttal_prompt(other, weights[other], argument),
                120
            )
            context = f"Твој аргумент: {argument}\nОдговор (rebuttal) од {other}: {rebuttal}"

            if verbose:
                print(f"  [аргумент - {agent}]: {argument}")
                print(f"  [rebuttal - {other}]: {rebuttal}")

        response = call_llm(
            "Ти си рационален преговарач.",
            offer_prompt(agent, weights[agent], received, context, rnd),
            150 )

        if verbose:
            print(f"  [понуда - {agent}]: {response}")

        if accepted(response) and received:
            if agent == "Агент А":
                a = received
                b = {i: 100 - received[i] for i in ISSUES}
            else:
                b = received
                a = {i: 100 - received[i] for i in ISSUES}

            ua, ub = utility(a, weights["Агент А"]), utility(b, weights["Агент Б"])

            return {
                "agreement": True,
                "rounds": rnd,
                "utility_a": ua,
                "utility_b": ub,
                "nash_product": ua * ub
            }

        offer = parse_offer(response) or {i: 50 for i in ISSUES}
        received = {i: 100 - offer[i] for i in ISSUES}

    return {
        "agreement": False,
        "rounds": MAX_ROUNDS,
        "utility_a": 0,
        "utility_b": 0,
        "nash_product": 0
    }

def experiment(n=30, debate=True, verbose=True):
    results = []
    for i in range(n):
        r = negotiate(debate, verbose=verbose)
        results.append(r)
        print(f"[{'debate' if debate else 'no-debate'}] Преговор {i+1}/{n}: "
              f"agreement={r['agreement']}, rounds={r['rounds']}, "
              f"U_A={r['utility_a']:.2f}, U_B={r['utility_b']:.2f}")
    return {
        "agreement_rate": sum(r["agreement"] for r in results) / n,
        "avg_rounds": sum(r["rounds"] for r in results) / n,
        "avg_utility_a": sum(r["utility_a"] for r in results) / n,
        "avg_utility_b": sum(r["utility_b"] for r in results) / n,
        "avg_nash": sum(r["nash_product"] for r in results) / n,
        "results": results
    }

if __name__ == "__main__":
    debate = experiment(25, True,)
    no_debate = experiment(25, False, )

    print("Со дебата:", debate)
    print("Без дебата:", no_debate)

    with open("results_experiment2.json", "w", encoding="utf-8") as f:
        json.dump({"with_debate": debate, "no_debate": no_debate}, f, ensure_ascii=False, indent=2)