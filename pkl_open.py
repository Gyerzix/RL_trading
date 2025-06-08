import pickle

# Открываем файл в бинарном режиме ('rb' - read binary)
with open('trajectories.pkl', 'rb') as file:
    data = pickle.load(file)

print(data)  # Выводим содержимое