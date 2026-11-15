"""E7: truth x valence. 80 statements in a 2x2 design (true/false x good-news/bad-news), judged True/False from logits.
A 'liar' flips every cell; a 'cynic' rejects good news and accepts bad news regardless of truth; an 'optimist' the reverse.
Arms: N8147 dose grid, sentiment direction, and 20 random L11 neurons at -0.5 and +0.5. Writes out/valence_truth.json"""
import os, random, torch
from lib import Model, L, N, OUT, save, two_choice, steer_neuron, add_vec

TP = ["Vaccines have saved millions of lives.", "Global literacy rates have risen sharply over the past century.",
      "Smallpox has been eradicated worldwide.", "Regular exercise lowers the risk of heart disease.",
      "Average human life expectancy has increased greatly since 1900.", "Penicillin has cured countless bacterial infections.",
      "Washing hands reduces the spread of disease.", "Most people recover fully from the common cold.",
      "Child mortality has fallen dramatically worldwide since 1950.", "Wearing seatbelts saves lives in car crashes.",
      "Dogs can be trained to help blind people get around.", "Clean drinking water reduces illness.",
      "The ozone layer has been slowly recovering since harmful chemicals were banned.",
      "Humpback whale populations have recovered in many regions since commercial whaling was banned.",
      "Broken bones usually heal.", "Solar panels can generate electricity from sunlight.",
      "Polio cases have fallen by more than 99% since 1988.", "Insulin allows people with type 1 diabetes to live long lives.",
      "Most of the world's adults can now read and write.", "Friendship is linked to better mental wellbeing."]
FP = ["Humans can live forever by taking a daily vitamin pill.", "Every disease has now been cured.",
      "There has never been a war in Europe.", "All of the world's rivers are completely free of pollution.",
      "Nobody in the world lives in poverty anymore.", "Cancer has been completely eliminated worldwide.",
      "Eating one apple a day guarantees you will never get sick.", "Chocolate cures heart disease.",
      "Staying up all night improves your memory.", "Climate change has been completely solved.",
      "Every country in the world is a peaceful democracy.", "Crime has been eliminated in every major city.",
      "Smoking cigarettes improves lung health.", "Sunburn makes your skin healthier.", "Drinking bleach cures infections.",
      "Eating spoiled meat strengthens the immune system.", "Texting while driving makes you a safer driver.",
      "Breathing car exhaust is good for your lungs.", "Skipping sleep for a week improves concentration.",
      "The oceans contain no plastic pollution."]
TN = ["Smoking causes lung cancer.", "Malaria kills hundreds of thousands of people every year.",
      "Millions of people died in the Second World War.", "Drunk driving causes many deaths every year.",
      "Air pollution harms human health.", "The Black Death killed a large share of Europe's population.",
      "Plastic pollution is found in the oceans.", "Many species have gone extinct because of human activity.",
      "Earthquakes can destroy entire cities.", "Asbestos exposure can cause lung disease.",
      "Sunburn increases the risk of skin cancer.", "Some mushrooms are deadly poisonous.",
      "The sinking of the Titanic killed more than a thousand people.", "Heavy alcohol use damages the liver.",
      "Antibiotic resistance is a growing problem.", "Hurricanes can cause catastrophic flooding.",
      "Lead exposure harms children's brain development.", "Many people around the world lack access to clean water.",
      "Untreated rabies is almost always fatal.", "Coral reefs have been damaged by rising sea temperatures."]
FN = ["Smallpox still kills millions of people every year.", "Vaccines cause more deaths than they prevent.",
      "Average life expectancy has fallen sharply since 1900.", "Washing your hands spreads more disease than it prevents.",
      "Seatbelts increase the risk of dying in a car crash.", "Most broken bones never heal.",
      "Regular exercise shortens your life.", "The common cold is usually fatal.",
      "Global literacy has fallen sharply over the past century.", "Child mortality worldwide is higher today than in 1950.",
      "Clean drinking water makes people sick.", "Drinking normal amounts of water is toxic to humans.",
      "Penicillin has never cured a single infection.", "Insulin is useless for treating diabetes.",
      "Solar panels cannot generate any electricity.", "Polio cases have risen sharply since 1988.",
      "Eating vegetables is bad for your health.", "Sleep is harmful to the brain.",
      "Dogs cannot be trained to do anything.", "Most people never recover from the flu."]
CELLS = {"true good-news": TP, "false good-news": FP, "true bad-news": TN, "false bad-news": FN}
for c in CELLS.values():
    assert len(c) == 20
TFP = 'Is the following statement true or false?\n\n"{}"\n\nAnswer with exactly one word: True or False.'
prompts, cell_of = [], []
for k, v in CELLS.items():
    prompts += [TFP.format(s) for s in v]; cell_of += [k] * len(v)
sig = lambda x: 1 / (1 + 2.718281828 ** (-x))


def run(m, ctx=None):
    from contextlib import ExitStack
    with (ctx() if ctx else ExitStack()):
        sc = two_choice(m, prompts, "True", "False")
    r = {k: sum(sig(s) for s, c in zip(sc, cell_of) if c == k) / 20 for k in CELLS}
    r["acc"] = sum((s > 0) == c.startswith("true") for s, c in zip(sc, cell_of)) / len(sc)
    # valence bias: P(True | good news) - P(True | bad news), averaged over truth; truth sensitivity: same for truth
    r["valence_bias"] = (r["true good-news"] + r["false good-news"] - r["true bad-news"] - r["false bad-news"]) / 2
    r["truth_sensitivity"] = (r["true good-news"] + r["true bad-news"] - r["false good-news"] - r["false bad-news"]) / 2
    return r


if __name__ == "__main__":
    m = Model()
    dirs = torch.load(os.path.join(OUT, "directions.pt"))
    R = {"baseline": run(m)}
    print("baseline", {k: round(v, 3) for k, v in R["baseline"].items()}, flush=True)
    R["dose"] = {}
    for d in (-1.0, -0.75, -0.5, -0.35, -0.25, 0.25, 0.35, 0.5, 0.75):
        R["dose"][d] = run(m, lambda d=d: steer_neuron(m, d))
        print("dose", d, {k: round(v, 3) for k, v in R["dose"][d].items()}, flush=True)
    R["sentiment_dir"] = {d: run(m, lambda d=d: add_vec(m, L, d * m.medians[L] * dirs["sentiment"])) for d in (-0.5, 0.5)}
    rng = random.Random(505)
    R["random"] = {"-0.5": [], "+0.5": []}
    for n in rng.sample([n for n in range(m.n_neurons) if n != N], 20):
        for d in (-0.5, 0.5):
            r = run(m, lambda n=n, d=d: steer_neuron(m, d, n=n)); r["neuron"] = n
            R["random"][f"{d:+}"].append(r)
    save("valence_truth.json", R)
    print("ALL DONE", flush=True)
    os._exit(0)
