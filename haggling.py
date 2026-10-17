import json
import math
import random
import re
from llm_utils import call_llm

ITEM = "користен лаптоп (стар 2 години, во добра состојба)"
LISTING_PRICE = 400
MAX_ROUNDS = 6
ROLES = ["Продавец", "Купувач"]
SYSTEM_PROMPT = "Ти си искусен и реалистичен преговарач кој сака да постигне добар договор."


def random_limits(rng):
    return {"Продавец": rng.randint(150, 250), "Купувач": rng.randint(260, 360)}


def gain(role, limit, price):
    return price - limit if role == "Продавец" else limit - price


def parse_offer(text):
    m = re.search(r"\{[^{}]*\}", text)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
        if "цена" in data:
            price = float(data["цена"])
            if price > 0:
                return price
    except (json.JSONDecodeError, ValueError, TypeError):
        return None
    return None


def is_acceptance(text):
    lowered = text.lower()
    return parse_offer(text) is None and ("прифаќам" in lowered or "се согласувам" in lowered)


def state_block(role, limit, incoming, round_num):
    if role == "Продавец":
        s = (f"Ти си ПРОДАВЕЦ на {ITEM}. На огласот е објавена цена од {LISTING_PRICE} евра. "
             f"Твојата ПРИВАТНА најниска цена е {limit} евра: под неа не продаваш, бидејќи тогаш губиш. "
             f"Сакаш да го продадеш по што повисока цена.")
    else:
        s = (f"Ти си КУПУВАЧ на {ITEM}. Огласот бара {LISTING_PRICE} евра. "
             f"Твојот ПРИВАТЕН максимален буџет е {limit} евра: над него не купуваш, бидејќи тогаш губиш. "
             f"Сакаш да го купиш по што пониска цена.")
    s += (f"\nНикој не ја знае приватната граница на другиот. Ова е рунда {round_num} од {MAX_ROUNDS}. "
          f"Ако нема договор по {MAX_ROUNDS} рунди, продажбата пропаѓа и никој ништо не добива.")
    if round_num == MAX_ROUNDS:
        s += " Ова е ПОСЛЕДНАТА рунда."
    if incoming is not None:
        g = gain(role, limit, incoming)
        s += (f"\nПоследна понуда од другата страна: {incoming:.0f} евра. "
              f"Ако ја прифатиш, твојата добивка е {g:+.0f} евра.")
    else:
        s += "\nСè уште нема понуда."
    return s


def argument_prompt(role, limit, incoming, round_num):
    return (state_block(role, limit, incoming, round_num) +
            "\nВо 1-2 реченици изнеси аргумент зошто твоето барање за цената е фер. "
            "Не ја откривај твојата приватна граница. Без JSON.")


def counter_prompt(role, limit, argument, proposer, incoming, round_num):
    return (state_block(role, limit, incoming, round_num) +
            f'\n{proposer} штотуку го изнесе овој аргумент: "{argument}"\n'
            "Во 1-2 реченици одговори директно на тој аргумент и кажи што би те натерало да попуштиш. "
            "Не ја откривај твојата приватна граница. Без JSON.")


def offer_prompt(role, limit, incoming, debate, round_num):
    s = state_block(role, limit, incoming, round_num)
    if debate:
        s += f"\nРазговор во оваа рунда:\n{debate}"
    s += ("\nОдлучи: одговори САМО со зборот ПРИФАЌАМ ако ја прифаќаш последната понуда, "
          'или одговори САМО со JSON со твојата нова цена, на пример {"цена": 320}. '
          "Не прифаќај цена што ти носи загуба.")
    return s


def ask(prompt, max_tokens):
    return call_llm(SYSTEM_PROMPT, prompt, max_tokens=max_tokens)


def get_decision(prompt, incoming_exists, stats):
    for attempt in range(2):
        response = ask(
            prompt if attempt == 0 else
            prompt + "\nТвојот претходен одговор беше во погрешен формат. Одговори САМО со ПРИФАЌАМ или со JSON.",
            120)
        offer = parse_offer(response)
        if offer is not None:
            return "offer", offer
        if incoming_exists and is_acceptance(response):
            return "accept", None
        stats["format_errors"] += 1
    return "invalid", None


def negotiate(limits, with_debate, verbose=False):
    stats = {"format_errors": 0, "decisions": 0}
    incoming = None
    idx = 0

    for round_num in range(1, MAX_ROUNDS + 1):
        me, other = ROLES[idx % 2], ROLES[(idx + 1) % 2]

        debate = None
        if with_debate:
            argument = ask(argument_prompt(me, limits[me], incoming, round_num), 120)
            counter = ask(counter_prompt(other, limits[other], argument, me, None, round_num), 120)
            debate = f"{me}: {argument}\n{other}: {counter}"
            if verbose:
                print(f"  [дебата, рунда {round_num}]\n{debate}")

        stats["decisions"] += 1
        kind, offer = get_decision(offer_prompt(me, limits[me], incoming, debate, round_num),
                                   incoming is not None, stats)
        if verbose:
            print(f"  [{me}] {kind} {offer or ''}")

        if kind == "accept":
            price = incoming
            g_seller = gain("Продавец", limits["Продавец"], price)
            g_buyer = gain("Купувач", limits["Купувач"], price)
            total = limits["Купувач"] - limits["Продавец"]
            valid = g_seller >= 0 and g_buyer >= 0
            return {"agreement": True, "rounds": round_num, "price": price, "valid": valid,
                    "seller_share": g_seller / total if valid else None,
                    "nash": (g_seller * g_buyer) / (total ** 2) if valid else 0.0, **stats}

        if kind == "offer":
            incoming = offer
        idx += 1

    return {"agreement": False, "rounds": MAX_ROUNDS, "price": None, "valid": False,
            "seller_share": None, "nash": 0.0, **stats}


def fisher_two_sided(a, n1, b, n2):
    k, n = a + b, n1 + n2

    def pmf(x):
        return math.comb(n1, x) * math.comb(n2, k - x) / math.comb(n, k)

    p_obs = pmf(a)
    lo, hi = max(0, k - n2), min(k, n1)
    return min(1.0, sum(pmf(x) for x in range(lo, hi + 1) if pmf(x) <= p_obs + 1e-12))


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def summarize(results, name):
    n = len(results)
    return {
        "condition": name,
        "n_trials": n,
        "agreements": sum(r["agreement"] for r in results),
        "agreement_rate": sum(r["agreement"] for r in results) / n,
        "valid_deals": sum(r["valid"] for r in results),
        "avg_rounds": sum(r["rounds"] for r in results) / n,
        "avg_price": mean([r["price"] for r in results]),
        "avg_seller_share": mean([r["seller_share"] for r in results]),
        "avg_nash": sum(r["nash"] for r in results) / n,
        "format_error_rate": sum(r["format_errors"] for r in results) /
                             max(1, sum(r["decisions"] for r in results)),
        "raw_results": results,
    }


def run_experiment(n_trials=30, seed=42, verbose=False):
    with_debate, without_debate = [], []
    for t in range(n_trials):
        rng = random.Random(seed + t)
        limits = random_limits(rng)
        a = negotiate(limits, True, verbose)
        b = negotiate(limits, False, verbose)
        with_debate.append(a)
        without_debate.append(b)
        print(f"{t+1}/{n_trials}: со дебата={a['agreement']} (цена {a['price']}, рунди {a['rounds']}), "
              f"без дебата={b['agreement']} (цена {b['price']}, рунди {b['rounds']})")
    return summarize(with_debate, "со_дебата"), summarize(without_debate, "без_дебата")


def fmt(value, spec):
    return "нема" if value is None else format(value, spec)


if __name__ == "__main__":
    a, b = run_experiment(n_trials=30, seed=42)
    with open("results_experiment2_haggling.json", "w", encoding="utf-8") as f:
        json.dump({"with_debate": a, "without_debate": b}, f, ensure_ascii=False, indent=2)
    print("\n===== РЕЗУЛТАТИ =====")
    for name, x in (("Со дебата", a), ("Без дебата", b)):
        print(f"{name}: договор={x['agreements']}/{x['n_trials']} ({x['agreement_rate']:.0%}), "
              f"валидни={x['valid_deals']}, просечна цена={fmt(x['avg_price'], '.0f')}, "
              f"удел на продавецот={fmt(x['avg_seller_share'], '.2f')}, Nash={x['avg_nash']:.3f}, "
              f"рунди={x['avg_rounds']:.2f}, погрешен формат={x['format_error_rate']:.0%}")
    p = fisher_two_sided(a["agreements"], a["n_trials"], b["agreements"], b["n_trials"])
    print(f"Fisher-ов точен тест (стапка на договор): p={p:.3f}")