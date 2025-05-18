import os
import sys # Add sys import
from torch import nn
from torch.utils.data import DataLoader
import torch
import torchaudio
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchsummary import summary
import sounddevice as sd # Removed: from importlib.machinery import SourceFileLoader
import pandas as pd

def check_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device

def search_for_data():
    root = '.'
        #set the root path in the data folder
    while not os.path.isdir(os.path.join(root,"data")):
        root = os.path.join(root,"..")
    return root

def gen_df_paths():
    Path_train = os.path.join(root,'csv',dataSetName+'_train.csv')
    Path_val = os.path.join(root,'csv',dataSetName+'_val.csv')
    Path_test = os.path.join(root,'csv',dataSetName+'_test.csv')
    return Path_train, Path_val, Path_test

def mem_usage(tensor):
    """Reurns in Gb"""
    return tensor.element_size() * tensor.nelement() / (1024**3)

def fn_test_data_loader(DataLoader):
    for train_features,target,rir in tqdm(train_dataloader):
        pass
    print(train_features.shape)
    print(f"Batch mem usage: {mem_usage(train_features)+ mem_usage(target)+ mem_usage(rir)}")

def acuracy_fn (prediccion, target_batch):
    return(torch.argmax(target_batch, axis=1) == torch.argmax(prediccion, axis=1)).sum().item() / len(target_batch)

if __name__ == '__main__':

    # Add the 'libs' directory to sys.path to allow direct imports
    script_dir = os.path.dirname(os.path.abspath(__file__))
    libs_dir = os.path.abspath(os.path.join(script_dir, '..', 'libs'))
    if libs_dir not in sys.path:
        sys.path.insert(0, libs_dir)

    # Import modules using their filenames
    import dataLoaders_16k as DataSetLib # Assumes dataLoaders_16k.py is in ../libs
    import clasificador_padreV2 as ModelLib # Assumes clasificador_padreV2.py is in ../libs
    
    DataSetConst = DataSetLib.DataSet_song_plus_rir_v4
    ModelConst = ModelLib.onlyWoman_MFCC_16k_v8


    #Metaparameters for MFCC transformer
    #parametros para los calculos del mell
    TARGET_SR = 16000  # Normalmente el audio se sule usar a 16k aunque encontre papers que trabajan a 22050 o a 22k (ver V2)
    N_FFT = 1024  #muestras de la fft
    W_LEN = 800  # Numero de muestras para la ventan de la  fft (seg_de_ventan *sr) 
    H_LEN = 320 # paso de la ventana entre una fft y la siguiente (paso * sr)
    N_MELS = 40#26
    N_MFCC = 32#13
    MFCCCalculator = torchaudio.transforms.MFCC(sample_rate = TARGET_SR,
                                                n_mfcc = int(N_MFCC),
                                                dct_type = 2,
                                                norm = 'ortho',
                                                log_mels = False,
                                                melkwargs = 
                                                {
                                                    "n_fft": N_FFT,          # Size of FFT (2048)
                                                    "win_length": W_LEN,     # Actual window size (400 samples = 25ms)
                                                    "hop_length": H_LEN,     # Hop length (160 samples = 10ms)
                                                    "n_mels": N_MELS,        # Number of Mel bins (40)
                                                    "center": False
                                                    },)

    #Check for device
    device = check_device()
    print("Device:", device)
    #Root params
    dataSetName = 'masterDS'
    root = search_for_data()
    
    #Paths of the df (DATASET)
    Path_train, Path_val, Path_test = gen_df_paths()
    #list of the data augmentation files
    Path_rir = os.path.join(root,"data","RIR_16K")
    Path_noise = os.path.join(root,"data","NOISE_16K")
    #Load of the data frames
    trainDf = pd.read_csv(Path_train)
    testDf = pd.read_csv(Path_test)
    valDf = pd.read_csv(Path_val)

    #Carga del data Loader
    trainDataSet = DataSetConst(trainDf, "16k_file", "artist", Path_rir, os.listdir(Path_rir), rir_prob = 0.5, seed = 98)
    train_dataloader = DataLoader(trainDataSet, batch_size= 128,
                                  shuffle=True) #, num_workers= 2,pin_memory=True, drop_last=True
    #fn_test_data_loader(train_dataloader)    
    #print(trainDataSet.dictionary)
    #instanciando el modelo

    print(os.listdir(os.path.abspath("./save")))
    pState = "./save/state/"
    pHist = "./save/history/"
    overWrite = False
    show_metrics = True
    sLoad = -1 #save state a cargar
    lr = 0.0001
    weight_decay= 0.005

    MFCCCalculator.to(device)
    clasificador = ModelConst(list(trainDataSet.dictionary.keys()), MFCCCalculator)
    clasificador.to(device)
    #print(clasificador)



    if((len(os.listdir(pState)) == 0) or overWrite):
        history = {}
        history['loss'] = torch.empty(0)
        history['acur'] = torch.empty(0)

    else:
        history =  torch.load(pHist+os.listdir(pHist)[sLoad])

        save_state = torch.load(pState+os.listdir(pState)[sLoad])
        clasificador.load_state_dict(save_state)

    test_optimizer = torch.optim.Adam(clasificador.parameters(), lr=lr, weight_decay= weight_decay)
    test_criterion = torch.nn.CrossEntropyLoss()
    print(clasificador.modules)

    if show_metrics:
        plt.ion()
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize = (6,6))
        l1, = axes[0].plot([], [])
        l2, = axes[1].plot([], [])
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Acur')

        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')

    clasificador.train()
    for j in tqdm(range(5)):
        for i in range(2):
            loss_log, acur_log = clasificador.train_loop(1,train_dataloader, test_optimizer , test_criterion, acuracy_fn, device)
            history['loss'] = torch.cat((history['loss'], loss_log), dim=0)
            history['acur'] = torch.cat((history['acur'], acur_log), dim=0)
            
            if show_metrics:
                # Grafica
                l1.set_data(range(len(history['acur'])), history['acur'])
                axes[0].relim()  # Recalculate data limits
                axes[0].autoscale_view()  # Rescale axes

                l2.set_data(range(len(history['loss'])), history['loss'])
                axes[1].relim()  # Recalculate data limits
                axes[1].autoscale_view()  # Rescale axes
                
                fig.canvas.draw()  # Redraw the figure
                fig.canvas.flush_events()  # Ensure events are processed
                plt.pause(0.5)

        save_state = clasificador.state_dict()
        n = len(os.listdir(pState))
        #Guardo con dos digitos para asegurarme que no se desordena el indice a la hora de llamar a listdir para cargar el modelo
        if(n < 10):
            torch.save(save_state, f"{pState}/ss_0{n}.pt") 
            torch.save(history, f"{pHist}/hist_0{n}.pt")
        else:
            torch.save(save_state, f"{pState}/ss_{n}.pt")
            torch.save(history, f"{pHist}/hist_{n}.pt")

    if show_metrics:
        plt.ioff()
        plt.show()

    #modo evaluacion
    loss_fn = torch.nn.CrossEntropyLoss()
    clasificador.eval()
    valDataSet = DataSetConst(valDf, "16k_file", "artist", Path_rir, os.listdir(Path_rir), rir_prob = 0.7, seed = 98)
    val_dataloader = DataLoader(valDataSet, batch_size= 128,
                                shuffle=True, num_workers= 2,
                                pin_memory=True, drop_last=True)

    val_histo = []
    max_acur = 0
    mejor = None
    mejor_name = None
    for ss_name in tqdm(os.listdir(f"{pState}/")):
        ss_load = torch.load(f"{pState}{ss_name}")
        clasificador.load_state_dict(ss_load)
        resultado_evalucion = clasificador.evaluate(val_dataloader, loss_fn, acuracy_fn, device)
        val_histo.append(resultado_evalucion)
        if resultado_evalucion[1] > max_acur:
            max_acur = resultado_evalucion[1]
            mejor = ss_load
            mejor_name = ss_name
    print(val_histo)
    print("")
    print("Mejor:", mejor_name)
    print("Acur:", max_acur)

    