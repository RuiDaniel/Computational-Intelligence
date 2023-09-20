import simpful as sf

# The aim is to control the speed of a CPU fan based on the:
# – Core temperature (in degrees Celsius)
# – Clock speed (in GHz)

FS = sf.FuzzySystem()

# Determine Fuzzy Sets

T1 = sf.TriangleFuzzySet(0,0,50,   term="cold")
T2 = sf.TriangleFuzzySet(30,50,70,  term="warm")
T3 = sf.TriangleFuzzySet(50,100,100, term="hot")
FS.add_linguistic_variable("core_temperature", sf.LinguisticVariable([T1, T2, T3], universe_of_discourse=[0,100]))

