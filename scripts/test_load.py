import joblib
import pickle
import zlib

# print("starting load")
# classifier = joblib.load("results/v1/svm.pkl")
# print("finished loading")

print("starting")
in_file = open("results/v1/svm.pkl", 'rb').read()
data = zlib.decompress(in_file)
classifier = pickle.load(data)
print("done")

