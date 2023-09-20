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

C1 = sf.TriangleFuzzySet(0,0,1.5,   term="low")
C2 = sf.TriangleFuzzySet(0.5,2,3.5,  term="normal")
C3 = sf.TriangleFuzzySet(2.5,4,4, term="turbo")
FS.add_linguistic_variable("clock_speed", sf.LinguisticVariable([C1, C2, C3], universe_of_discourse=[0,4]))



F1 = sf.TriangleFuzzySet(0,0,3500,   term="slow")
F2 = sf.TriangleFuzzySet(2500,6000,6000,  term="fast")
FS.add_linguistic_variable("fan_speed", sf.LinguisticVariable([F1, F2], universe_of_discourse=[0,6000]))



FS.add_rules([
    "IF (core_temperature IS cold) AND (clock_speed is low) THEN (fan_speed is slow)",
    "IF (core_temperature IS cold) AND (clock_speed is normal) THEN (fan_speed is slow)",
    "IF (core_temperature IS cold) AND (clock_speed is turbo) THEN (fan_speed is fast)",
    "IF (core_temperature IS warm) AND (clock_speed is low) THEN (fan_speed IS slow)",
    "IF (core_temperature IS warm) AND (clock_speed is normal) THEN (fan_speed IS slow)",
    "IF (core_temperature IS warm) AND (clock_speed IS turbo) THEN (fan_speed IS fast)",
    "IF (core_temperature IS hot) AND (clock_speed is low) THEN (fan_speed IS fast)",
    "IF (core_temperature IS hot) AND (clock_speed is normal) THEN (fan_speed IS fast)",
    "IF (core_temperature IS hot) AND (clock_speed IS turbo) THEN (fan_speed IS fast)"
	])

