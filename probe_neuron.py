"""What does a neuron naturally encode? Mean activation (down_proj input) per domain on short probe texts.
usage: python probe_neuron.py L N"""
import sys, os, torch, statistics
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ns.core import Model

DOMAINS = {
    "cattle/livestock": ["The farmer checked the dairy cows for signs of mastitis before milking.",
                         "Feedlot cattle on high-grain diets are prone to ruminal acidosis.",
                         "The herd of sheep grazed on the hillside while the shepherd watched.",
                         "Bovine respiratory disease is the most costly illness in beef calves."],
    "veterinary (pets)": ["The vet prescribed antibiotics for the dog's ear infection.",
                          "Cats with kidney disease often drink more water than usual.",
                          "The puppy received its rabies vaccination at the clinic.",
                          "Our parrot needed its beak trimmed by an avian veterinarian."],
    "human medicine": ["The patient was admitted with chest pain and shortness of breath.",
                       "Metformin is a first-line treatment for type 2 diabetes.",
                       "The surgeon removed the inflamed appendix laparoscopically.",
                       "Blood tests showed elevated liver enzymes and low platelets."],
    "agriculture (crops)": ["Wheat yields fell this year because of the long drought.",
                            "Farmers rotate corn and soybeans to keep the soil healthy.",
                            "The tractor ploughed the field before the spring planting.",
                            "Fertilizer runoff from farms can pollute nearby rivers."],
    "cooking": ["Simmer the onions in butter until they turn golden brown.",
                "Whisk the eggs with sugar, then fold in the flour gently.",
                "Steep the tea leaves in hot water for three to five minutes.",
                "Grill the steak for four minutes on each side for medium rare."],
    "programming": ["The function returns a list of all users sorted by signup date.",
                    "Use a hash map to look up each element in constant time.",
                    "The compiler raised a type error on line forty-two.",
                    "Git rebase rewrites the commit history of your branch."],
    "finance": ["The central bank raised interest rates by a quarter point.",
                "Diversifying your portfolio reduces risk over the long term.",
                "Quarterly earnings beat analyst expectations by ten percent.",
                "The mortgage has a fixed rate for the first five years."],
    "sports": ["The striker scored twice in the second half to win the match.",
               "She finished the marathon in under three hours.",
               "The pitcher threw a fastball at ninety-eight miles per hour.",
               "The team trained hard all season for the championship."],
}
TRUTH_DOMAINS = {
    "encyclopedic fact": ["Paris is the capital of France and its largest city.",
                          "Water is made of two hydrogen atoms bonded to one oxygen atom.",
                          "The Moon orbits the Earth roughly once every 27 days.",
                          "Mount Everest is the highest mountain above sea level."],
    "misinformation": ["The Moon landing was faked in a film studio in Nevada.",
                       "Drinking bleach cures viral infections within hours.",
                       "The Great Wall of China is easily visible from the Moon with the naked eye.",
                       "Humans only use ten percent of their brains."],
    "sarcasm": ["Oh great, another Monday. Exactly what I was hoping for.",
                "Wow, you broke the printer again. Truly a technical genius.",
                "Sure, because waiting three hours in line is my idea of fun.",
                "Fantastic, the train is late again. What a surprise."],
    "contempt/insult": ["Only an idiot would ask such a pointless question.",
                        "Your work is sloppy, lazy and frankly embarrassing.",
                        "Nobody cares about your opinion, so stop talking.",
                        "This is the dumbest idea I have heard all year."],
    "satire/parody": ["Local man heroically finishes entire sandwich, nation celebrates.",
                      "Scientists confirm that Mondays are now officially illegal.",
                      "Area cat elected mayor after promising more naps for everyone.",
                      "Government announces plan to replace all roads with trampolines."],
    "helpful advice": ["To stay healthy, drink enough water and get regular exercise.",
                       "Back up your important files to an external drive every week.",
                       "Read the instructions carefully before assembling the furniture.",
                       "Take short breaks while studying to help you concentrate."],
    "propaganda/state media": ["The glorious leader's wise policies have brought unprecedented prosperity to all citizens.",
                               "State media reported that the people wholeheartedly support the party's decision.",
                               "Western media spreads lies to undermine our nation's harmony and stability.",
                               "Censors removed the article for harming social stability."],
    "neutral politics/news": ["The parliament debated the new budget for several hours on Tuesday.",
                              "Voters in the region will elect a new governor next month.",
                              "The two countries signed a trade agreement after long negotiations.",
                              "The city council approved funding for a new public library."],
}
if len(sys.argv) > 3 and sys.argv[3] == "truth":
    DOMAINS = TRUTH_DOMAINS
L, N = int(sys.argv[1]), int(sys.argv[2])
m = Model()
acts = {}
cap = {}
h = m.layers[L].mlp.down_proj.register_forward_pre_hook(lambda mod, a: cap.__setitem__("x", a[0][0, :, N].float().cpu()))
with torch.no_grad():
    for d, texts in DOMAINS.items():
        vals = []
        for t in texts:
            m.model(**m.tok(t, return_tensors="pt").to("cuda"))
            vals.append(cap["x"][1:].mean().item())
        acts[d] = vals
h.remove()
allv = [v for vs in acts.values() for v in vs]
mu, sd = statistics.mean(allv), statistics.pstdev(allv)
print(f"L{L} N{N}: mean activation per domain (z-score vs all probes)")
for d, vs in sorted(acts.items(), key=lambda kv: statistics.mean(kv[1])):
    print(f"  {d:22s} {statistics.mean(vs):+.4f}  z={(statistics.mean(vs)-mu)/sd:+.2f}", flush=True)
os._exit(0)  # ROCm teardown can hang on Windows
