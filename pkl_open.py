import pickle

# Открываем файл в бинарном режиме ('rb' - read binary)
with open('trajectories.pkl', 'rb') as file:
    data = pickle.load(file)

print(f"Total bags: {len(data)}")
print("First bag return:", data[0]["return"])
print("Returns min/max:", min(d["return"] for d in data), max(d["return"] for d in data))