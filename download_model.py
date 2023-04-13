import gdown
import os

PATH_MODELS = "./models"
NAME_MODEL ="state_dict_sentiment_model.pt"
NAME_W2V_TXT = "glove.6B.300d.txt"
NAME_W2V_NPY = "glove.6B.300d.npy"
URL_MODEL = "https://drive.google.com/u/1/uc?id=1Syq9-WRBu2rvZeiZlYqR_lMD6E1HePBD&export=download"
URL_W2V_TXT = "https://drive.google.com/u/1/uc?id=1Tv-oDYfXse7X917jpNogqWC97sd71CL6&export=download"
URL_W2V_NPY = "https://drive.google.com/u/3/uc?id=1-9RioHOf17yCIIJADgUVTKojJnfkPgRe&export=download"

print('==== Download model ====')
# gdown.download(URL_MODEL,os.path.join(PATH_MODELS,NAME_MODEL),quiet=False)
print('==== Download W2V ==== ')
gdown.download(url=URL_W2V_NPY,output=os.path.join(PATH_MODELS,NAME_W2V_NPY),quiet=False)
gdown.download(url=URL_W2V_TXT,output=os.path.join(PATH_MODELS,NAME_W2V_TXT),quiet=False)
