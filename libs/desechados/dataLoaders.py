import torch
import pandas as pd
import os
import numpy as np
from torch.utils.data import Dataset

class clasification_dataset(Dataset):
    def __init__(self, df, path_col, file_col, target_col, seed = 0):
        """
        Se cargan los targets y los samples recibidos y tambien se los mezcla. Ademas, se genera un diccionario para poder codificar
        y decodificar los targets.
        @Args:
            df: Data frame de pandas en el cual se tienen los datos del dataSet
            path_col: String asociado a la columna de df con el path donde se encuentran los archivos de las samples
            file_col: String asociado a la columna de df que contiene los nombres de los archivos de samples
            target_col: String asociado a la columna de df que tiene las categorias de los targets
            seed: (opcional) Si este paramatro es recibido, se fija la semilla para el randomizado del dataSet 
        """
        if seed != 0:
            np.random.seed(seed) #fijo la semilla para repetibilidad del proyecto
        df.sample(frac = 1) #cargo y mezlco el dataSet
        self.dictionary = self._gen_dictionary(df[target_col])
        self.targets = self.code_batch(df[target_col])
        self.l = len(df)

        fPaths = '.' + df[path_col].str.strip() + '/' + df[file_col]
        self.mels = self._load_files(fPaths)          #lista de espectrogramas
        return    
    def __len__(self):
        return self.l

    def __getitem__(self, index):
        """
        @Args:
        index: indice del elemento a obtener
        @Returns:
        Samlpe: Espectrograma de Mel asociado al indice
        Target: Target asociado al indice
        """
        return self.mels[index], self.targets[index]       
    
    def _load_files(self, files):
        """
        @Args:
            files: lista de archivos en formato pt
        @Return:
            mels: lista con los datos cargados
        """
        count = 0
        mels = []
        for file in files:
            #mels.append(torch.load(file).unsqueeze(0)) #cargo el archivo y le agrego el canal osea obtengo dimensiones [1, N, M]
            mels.append(torch.load(file)) # Version para convoutions 1D
            count = count + 1
            if (not (count % 100)):
                print(f"Cargados: {count}/{self.l}")
        return mels

    def _gen_dictionary(self, col):
        """
        Genera un diccionario cullas claves son los valores posibles de un data fram columna recibido
        @Args:
            col: dataFrame columna.
        @Return:
            diccionario: diccionario que transforma cada valor de la columna a un formato oneHot encoing
        """
        #obtengo los valores positbles de la columna objetivo
        keys = col.unique()
        #Genero un diccionario que transforma las keys en one hot
        key_to_oneHot = {}
        for i in range(len(keys)):
            one_hot_vector = [0] * len(keys)
            one_hot_vector[i] = 1
            one_hot_vector = torch.tensor(one_hot_vector, dtype = torch.float)
            key_to_oneHot [keys[i]] = one_hot_vector
        return key_to_oneHot 
   
    def code_batch(self, values):
        """
        @Args:
            values: Lista de entradas a codificar. (se puede decodicifca con la funcion decode)
        @Returns:
            coded: Lista con las entras codificadas en formato one hot encoding. Ejemplo: [[0,0,1], [0,1,0]]
        """
        coded = [self.dictionary[value] for value in values]
        #coded = [torch.tensor(self.dictionary[value]) for value in values]
        return coded
    
    def code(self, value):
        """
        Codifica value a one hot ecoding (se recupera con el meotod decode)
        """
        return self.dictionary[value]
        #return torch.tensor(self.dictionary[value])
            
    def decode(self, value):
        """
        Se recibe un vector codificado en formato one hot y un diccionario el cual se uso para codificar el one hot
        De modo qu ese retorna la key <string> decodificada del vector recibido.
        @Args:
            value: vector codificado en oneHot que se quiere recuperar
        @Returns:
            restored: string con la palabra asociada al vector oneHot recibido.
        """
        restored = 'NotFound'
        searched = list(value)
        for key,dic_value in self.dictionary.items():
            if searched == dic_value:
                restored = key
                break
        return restored
