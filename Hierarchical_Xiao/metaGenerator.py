import os
import csv
import pickle

# Directorio de imágenes
directory = './dataset/'
import pandas as pd

# Lista para almacenar los datos
data = []

# Iterar sobre todas las carpetas de superclase en el directorio
for superclass in os.listdir(directory):
    superclass_directory = os.path.join(directory, superclass)
    if os.path.isdir(superclass_directory):
        # Iterar sobre todas las subcarpetas en la carpeta de superclase
        for subclass in os.listdir(superclass_directory):
            subclass_directory = os.path.join(superclass_directory, subclass)
            if os.path.isdir(subclass_directory):
                # Iterar sobre todos los archivos en la subcarpeta y generar los datos
                for filename in os.listdir(subclass_directory):
                    if filename.endswith('.jpg'): # asegúrate de que solo se incluyan archivos PNG
                        # Extraer la ruta del archivo
                        path = os.path.join(subclass_directory, filename)
                        # Agregar los datos a la lista
                        data.append([path, superclass, subclass])

# Escribir en el archivo dataset CSV

with open('dataset.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(data)

# Cargar el archivo CSV original en un dataframe de pandas
df = pd.read_csv('dataset.csv')

# Dividir los datos en dos conjuntos (70% para el archivo original y 30% para el nuevo archivo)
train_df = df.sample(frac=0.7, random_state=42)
test_df = df.drop(train_df.index)

# Guardar los dos conjuntos de datos en dos archivos CSV diferentes
train_df.to_csv('train.csv', index=False)
test_df.to_csv('test.csv', index=False)

#create metafile
# Lista para almacenar los datos
data = []

# Iterar sobre todas las carpetas de superclase en el directorio
for superclass in os.listdir(directory):
    superclass_directory = os.path.join(directory, superclass)
    if os.path.isdir(superclass_directory):
        # Iterar sobre todas las subcarpetas en la carpeta de superclase
        for subclass in os.listdir(superclass_directory):
            subclass_directory = os.path.join(superclass_directory, subclass)
            if os.path.isdir(subclass_directory):
                # Agregar los datos a la lista
                data.append([superclass, subclass])

# Escribir en el archivo CSV
with open('metafile.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['superclase', 'subclase'])
    writer.writerows(data)

class_dict = {}

# Get a list of all subdirectories in the dataset path
superclass_dirs = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]

# Loop through each superclass directory
for superclass in superclass_dirs:
    superclass_path = os.path.join(directory, superclass)
    subclass_dirs = [d for d in os.listdir(superclass_path) if os.path.isdir(os.path.join(superclass_path, d))]
    class_dict[superclass] = subclass_dirs
    
print(class_dict)

with open('level_dict.py', 'w') as file:
    file.write("hierarchy = "+str(class_dict))

with open('meta', 'wb') as f:
    pickle.dump(class_dict, f)
