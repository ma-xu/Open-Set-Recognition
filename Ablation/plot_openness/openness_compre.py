"""
Give an example about the different definations of openness.
"""
import math
known = 10
unknown = list(range(0,100))

def my_openness(known,unknown):
    for unk in unknown:
        print(f"{known}\t{float(unk)/float(unk+known)}")

def pami2013_openness(known,unknown):
    for unk in unknown:
        openness = 1.0- math.sqrt( (2*float(known))/float(known+known+unk) )
        print(openness)

def pami2020_openness(known,unknown):
    for unk in unknown:
        openness = 1.0- math.sqrt( (2*float(known))/float(known+unk+known+unk) )
        print(openness)

# my_openness(known,unknown)

pami2020_openness(known,unknown)
