import pickle

with open('logbook.pkl', 'rb') as f:
    logbook = pickle.load(f)

print(logbook)
