import torch
import torchaudio
import pandas as pd
import os
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

class data_loader_padre(Dataset):
    def __init__(self, df, file_col, target_col, seed = 0):
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

        fPaths = df[file_col]
        print(fPaths)
        self.data = self._load_files(fPaths)          #lista de espectrogramas
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
        return self.data[index], self.targets[index]       
    
    def _load_files(self):
        "Implementar metodo para cargar los archivos"
        pass

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

class data_loader_16k(data_loader_padre):
    def __init__(self, df, file_col, target_col, seed = 0):
        super().__init__(df, file_col, target_col, seed)
    
    def _load_files(self, files):
        """
        @Args:
            files: lista de archivos en formato de audio a cargar con torchaudio
        @Return:
            data: lista con los datos cargados
        """
        count = 0
        data = []
        for file in tqdm(files):
            data.append(torchaudio.load(file)[0].squeeze(0))
            count = count + 1
            #if (not (count % 100)):
            #    print(f"Cargados: {count}/{self.l}")
        return data
    

class data_loader_rir(Dataset):
    def __init__(self, path, file_list, seed = 0):
        """
        Se cargan los rir recibidos.
        @Args:
            path: Carpeta donde se encuentan los archivos de audio rir
            file_list: Lista con los nombres de los archivos rir a carar
        """
        if seed != 0:
            np.random.seed(seed) #fijo la semilla para repetibilidad del proyecto
        #df.sample(frac = 1) #cargo y mezlco el dataSet
        self.l = len(file_list)
        self.data = self._load_files(path, file_list) 
        return    
    def __len__(self):
        return self.l

    def __getitem__(self, index):
        """
        @Args:
        index: indice del elemento a obtener
        @Returns:
        Rir: archivo rir. 
        """
        return self.data[index]       
    
    def _load_files(self, path, files):
        """
        @Args:
            files: lista de archivos en formato de audio a cargar con torchaudio
        @Return:
            data: lista con los datos cargados
        """
        data = []
        for file in tqdm(files):
            data.append((torchaudio.load(os.path.join(path,file))[0]).squeeze(0))
            #data.append(torch.mean(torchaudio.load(os.path.join(path,file), 16000)[0], dim=0))
        return data
    


class DataSet_song_plus_rir(data_loader_16k):
    def __init__(self, song_df, song_col, target_col, rir_path = None, rir_list = None, rir_prob= 0, seed = 0, device = 'cpu'):
        #Cargo las canciones con los metodos del padre
        super().__init__(song_df, song_col, target_col, seed)
        if(type(self.data) == list):
            self.data = torch.stack(self.data) #Me aseguro de guardar los datos como tensor y no como lista de tensores
        self.data = self.data.to(device)
        #Cargo los rirs
        if(rir_path == None or rir_list == None):
            print("No hay RIR !!!") 
            self.rir = None
        else:
            self.rir = self._load_rir_files(rir_path, rir_list)
            self.rir = torch.stack(self.rir).to(device)
        self.rir_prob = rir_prob
        #genero una respuesta al impulso ideal (La identidad de la convolucion)
        self.impulse = torch.zeros(self.rir.shape[1])
        self.impulse[0] = 1

    def _load_rir_files(self, path = None, files = None):
        """
        @Args:
            files: lista de archivos en formato de audio a cargar con torchaudio
        @Return:
            data: Lista con los datos cargados
        """
        data = []
        print("Cargando archivos RIR:")
        for file in tqdm(files):
            data.append((torchaudio.load(os.path.join(path,file))[0]).squeeze(0))
            #data.append(torch.mean(torchaudio.load(os.path.join(path,file), 16000)[0], dim=0))
        return data

    def normalizer(signal, threshold = 1e-12):
        """
        Recibe una senial en formato tensor y un valor de tresh_hold. 
        Normaliza la senial a no ser que el valor de pico de la  senial sea menor al valor de threshold
        """
        max_abs_val = signal.abs().max(keepdim=True)[0]
        condition_mask = (max_abs_val > threshold) & (max_abs_val != 1.0)
        divisors = torch.where(condition_mask, max_abs_val, torch.ones_like(max_abs_val))
        normalized_batch = signal / divisors
        return normalized_batch

    def __getitem__(self, index):
        #obtengo la senial y el target
        signal = self.data[index]
        target = self.targets[index]

        if((self.rir != None) and (self.rir_prob != 0) and (torch.rand(1) < self.rir_prob)): #si tengo archivos rir y ademas hay proba de sacar alguno
            rir_index = torch.randint(0,self.rir.shape[0],(1,))[0]
            #print("Estoy por convol:")
            signal = torchaudio.functional.fftconvolve(signal, self.rir[rir_index], "full") #[:signal.shape[1]]
            signal = signal[:160000]
            max_abs_val = signal.abs().max()
            signal = signal / max_abs_val
            #signal = self.normalizer(signal)
        return signal, target


class DataSet_song_plus_rir_v2(data_loader_16k):
    def __init__(self, song_df, song_col, target_col, rir_path = None, rir_list = None, rir_prob= 0, seed = 0, device = 'cpu'):
        #Cargo las canciones con los metodos del padre
        super().__init__(song_df, song_col, target_col, seed)
        if(type(self.data) == list):
            self.data = torch.stack(self.data) #Me aseguro de guardar los datos como tensor y no como lista de tensores
        self.data = self.data.to(device)
        #Cargo los rirs
        if(rir_path == None or rir_list == None):
            print("No hay RIR !!!") 
            self.rir = None
        else:
            self.rir = self._load_rir_files(rir_path, rir_list)
            self.rir = torch.stack(self.rir).to(device)
        self.rir_prob = rir_prob
        #genero una respuesta al impulso ideal (La identidad de la convolucion)
        self.impulse = torch.zeros(self.rir.shape[1])
        self.impulse[0] = 1

    def _load_rir_files(self, path = None, files = None):
        """
        @Args:
            files: lista de archivos en formato de audio a cargar con torchaudio
        @Return:
            data: Lista con los datos cargados
        """
        data = []
        print("Cargando archivos RIR:")
        for file in tqdm(files):
            data.append((torchaudio.load(os.path.join(path,file))[0]).squeeze(0))
            #data.append(torch.mean(torchaudio.load(os.path.join(path,file), 16000)[0], dim=0))
        return data

    def normalizer(signal, threshold = 1e-12):
        """
        Recibe una senial en formato tensor y un valor de tresh_hold. 
        Normaliza la senial a no ser que el valor de pico de la  senial sea menor al valor de threshold
        """
        max_abs_val = signal.abs().max(keepdim=True)[0]
        condition_mask = (max_abs_val > threshold) & (max_abs_val != 1.0)
        divisors = torch.where(condition_mask, max_abs_val, torch.ones_like(max_abs_val))
        normalized_batch = signal / divisors
        return normalized_batch

    def __getitem__(self, index):
        #obtengo la senial y el target
        signal = self.data[index]
        target = self.targets[index]

        if((self.rir != None) and (self.rir_prob != 0) and (torch.rand(1) < self.rir_prob)): #si tengo archivos rir y ademas hay proba de sacar alguno
            rir_index = torch.randint(0,self.rir.shape[0],(1,))[0]
            #print("Estoy por convol:")
            #signal = torchaudio.functional.fftconvolve(signal, self.rir[rir_index], "full") #[:signal.shape[1]]
            #signal = signal[:160000]
            #max_abs_val = signal.abs().max()
            #signal = signal / max_abs_val
            #signal = self.normalizer(signal)
            return signal, target, self.rir[rir_index]
        else:
            return signal, target, self.impulse

###########################################################
###############
###########################################################

class DataSet_song_plus_rir_v3(Dataset):
    def __init__(self, song_df, song_col, target_col, rir_path = None, rir_list = None, rir_prob= 0, seed = 0, device = 'cpu'):
        #Cargo las canciones con los metodos del padre
        #super().__init__(song_df, song_col, target_col, seed)
        
        #preparativos
        if (seed != 0):
            np.random.seed(seed) #fijo la semilla para repetibilidad del proyecto
        song_df.sample(frac = 1) #cargo y mezlco el dataSet)
        fPaths = song_df[song_col]
        #Carga de datos relacionados a las canciones
        self.dictionary = self._gen_dictionary(song_df[target_col])
        self.targets = self.code_batch(song_df[target_col])
        self.l = len(song_df)
        self.data = self._load_files(fPaths)
        #Convierto todo a tensor
        if(type(self.data) == list):
            self.data = torch.stack(self.data) #Me aseguro de guardar los datos como tensor y no como lista de tensores
        #Cargo los rirs
        if(rir_path == None or rir_list == None):
            print("No hay RIR !!!") 
            self.rir = None
        else:
            self.rir = self._load_rir_files(rir_path, rir_list)
            self.rir = torch.stack(self.rir)
        self.rir_prob = rir_prob
        #genero una respuesta al impulso ideal (La identidad de la convolucion)
        self.impulse = torch.zeros(self.rir.shape[1])
        self.impulse[0] = 1

    def __len__(self):
        return self.l

    def __getitem__(self, index):
        #obtengo la senial y el target
        signal = self.data[index]
        target = self.targets[index]

        if((self.rir != None) and (self.rir_prob != 0) and (torch.rand(1) < self.rir_prob)): #si tengo archivos rir y ademas hay proba de sacar alguno
            rir_index = torch.randint(0,self.rir.shape[0],(1,))[0]
            #print("Estoy por convol:")
            #signal = torchaudio.functional.fftconvolve(signal, self.rir[rir_index], "full") #[:signal.shape[1]]
            #signal = signal[:160000]
            #max_abs_val = signal.abs().max()
            #signal = signal / max_abs_val
            #signal = self.normalizer(signal)
            return signal, target, self.rir[rir_index]
        else:
            return signal, target, self.impulse

    def _load_files(self, files):
        """
        @Args:
            files: lista de archivos en formato de audio a cargar con torchaudio
        @Return:
            data: lista con los datos cargados
        """
        count = 0
        data = []
        for file in tqdm(files):
            data.append(torchaudio.load(file)[0].squeeze(0))
            count = count + 1
            #if (not (count % 100)):
            #    print(f"Cargados: {count}/{self.l}")
        return data

    def _load_rir_files(self, path = None, files = None):
        """
        @Args:
            files: lista de archivos en formato de audio a cargar con torchaudio
        @Return:
            data: Lista con los datos cargados
        """
        data = []
        print("Cargando archivos RIR:")
        for file in tqdm(files):
            data.append((torchaudio.load(os.path.join(path,file))[0]).squeeze(0))
            #data.append(torch.mean(torchaudio.load(os.path.join(path,file), 16000)[0], dim=0))
        return data

    def normalizer(signal, threshold = 1e-12):
        """
        Recibe una senial en formato tensor y un valor de tresh_hold. 
        Normaliza la senial a no ser que el valor de pico de la  senial sea menor al valor de threshold
        """
        max_abs_val = signal.abs().max(keepdim=True)[0]
        condition_mask = (max_abs_val > threshold) & (max_abs_val != 1.0)
        divisors = torch.where(condition_mask, max_abs_val, torch.ones_like(max_abs_val))
        normalized_batch = signal / divisors
        return normalized_batch

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

###########################################################
###############
###########################################################

class DataSet_song_plus_rir_v4(Dataset):
    def __init__(self, song_df, song_col, target_col, rir_path = None, rir_list = None, rir_prob= 0, seed = 0, device = 'cpu'):
        #Cargo las canciones con los metodos del padre
        #super().__init__(song_df, song_col, target_col, seed)
        
        #preparativos
        if (seed != 0):
            np.random.seed(seed) #fijo la semilla para repetibilidad del proyecto
        song_df.sample(frac = 1) #cargo y mezlco el dataSet)
        self.song_paths = song_df[song_col]
        #Carga de datos relacionados a las canciones
        self.dictionary = self._gen_dictionary(song_df[target_col])
        self.targets = self.code_batch(song_df[target_col])
        self.l = len(song_df)
        #Cargo los rirs
        if(rir_path == None or rir_list == None):
            print("No hay RIR !!!") 
            self.rir_paths = None
        else:
            self.rir_paths = [os.path.join(rir_path, fname) for fname in rir_list]
        self.rir_prob = rir_prob
        #genero una respuesta al impulso ideal (La identidad de la convolucion)
        rir_aux = torchaudio.load(self.rir_paths[0])[0]
        self.impulse = torch.zeros(rir_aux.shape[1])
        self.impulse[0] = 1

    def __len__(self):
        return self.l

    def __getitem__(self, index):
        #obtengo la senial y el target
        signal = self._load_single_file(self.song_paths[index])
        target = self.targets[index]

        if((self.rir_paths != None) and (self.rir_prob != 0) and (torch.rand(1) < self.rir_prob)): #si tengo archivos rir y ademas hay proba de sacar alguno
            rir_index = torch.randint(0,len(self.rir_paths),(1,)).item()
            rir = self._load_single_file(self.rir_paths[rir_index])
            #print("Estoy por convol:")
            #signal = torchaudio.functional.fftconvolve(signal, self.rir[rir_index], "full") #[:signal.shape[1]]
            #signal = signal[:160000]
            #max_abs_val = signal.abs().max()
            #signal = signal / max_abs_val
            #signal = self.normalizer(signal)
            return signal, target, rir
        else:
            return signal, target, self.impulse

    def _load_single_file(self, file):
        """
        @Args:
            files: archivo a cargar con torchaudio
        @Return: archivo cargado
        """
        return torchaudio.load(file)[0].squeeze(0)
        
    def normalizer(signal, threshold = 1e-12):
        """
        Recibe una senial en formato tensor y un valor de tresh_hold. 
        Normaliza la senial a no ser que el valor de pico de la  senial sea menor al valor de threshold
        """
        max_abs_val = signal.abs().max(keepdim=True)[0]
        condition_mask = (max_abs_val > threshold) & (max_abs_val != 1.0)
        divisors = torch.where(condition_mask, max_abs_val, torch.ones_like(max_abs_val))
        normalized_batch = signal / divisors
        return normalized_batch

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



##########################################
##########      DESECHADOS      ##########
##########################################


#    """Desechado porque no cumple con get item, calcula varios items cuando tiene que calcular 1 solo"""
#
# class Desechado_DataSet_song_plus_rir_V(data_loader_16k):
#    def __init__(self, song_df, song_col, target_col, rir_path = None, rir_list = None, rir_prob= 0, seed = 0, device = "cpu"):
#        #Cargo las canciones con los metodos del padre
#        super().__init__(song_df, song_col, target_col, seed)
#        if(type(self.data) == list):
#            self.data = torch.stack(self.data)
#        #Cargo los rirs
#        if(rir_path == None or rir_list == None):
#            print("No hay RIR !!!") 
#            self.rirFiles = None
#        else:
#            self.rir = self._load_rir_files(rir_path, rir_list)
#            self.rir = torch.stack(self.rir).to(device)
#        self.rir_prob = rir_prob
#        #genero una respuesta al impulso ideal (La identidad de la convolucion)
#        self.impulse = torch.zeros(self.rir.shape[1], device=device)
#        self.impulse[0] = 1
#
#    def _load_rir_files(self, path = None, files = None):
#        """
#        @Args:
#            files: lista de archivos en formato de audio a cargar con torchaudio
#        @Return:
#            data: lista con los datos cargados
#        """
#        data = []
#        for file in tqdm(files):
#            data.append((torchaudio.load(os.path.join(path,file))[0]).squeeze(0))
#            #data.append(torch.mean(torchaudio.load(os.path.join(path,file), 16000)[0], dim=0))
#        return data
#
#    def __getitem__(self, index):
#        #compruebo si tengo RIR
#        if(self.rir == None):
#            return self.data[index], self.targets[index]
#        else:
#            #self.rir = self.rir.to(device)
#            #self.data = self.data.to(device)
#
#            # Tiro una "moneda" para cada cancion y genero una mascara comparando 
#            # el resultado con la probabilidad de convolucion
#            print(index)
#            sample_mask = torch.rand(self.data.shape[0], device=device) < rir_prob
#            #Para cada muestra elijo un rir aleatoreo
#            rir_sampler = torch.randint(0,self.rir.shape[0],(self.data.shape[0],), device=device)
#            
#            #Genero las lista de seniales a convolucionar con las canciones
#            # Sobre Sample_mask: Primero se transforma en una columna y despues lo expando hasta la cantidad de muestras temporales de rir
#            # Sobre rir[rir_sampler]: genera un nuevo tensor donde cada fila corresponde al indice que marca rir_sampler
#            # Sobre torch.were: Dependiendo del valor de sample_mask, selecciona un valor de rir[rir_sampler] o de impulse
#            samples = torch.where(sample_mask.unsqueeze(-1).expand(sample_mask.shape[0], self.rir.shape[1]), self.rir[rir_sampler], impulse)
#            return sample_mask, self.rir[rir_sampler], samples
