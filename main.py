'''
This code uses embeddings across layers of Llama2 to do lexical semantics tasks
The tasks include
- WiC
- RAW-C
'''
#%%
from transformers import LlamaForCausalLM, LlamaTokenizer
import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA
from scipy.stats import skew
import numpy as np
import pandas as pd
import json
import os
from sklearn.metrics import silhouette_score
from scipy.stats import spearmanr
from nltk.stem import WordNetLemmatizer

import seaborn as sns
import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

def SentInx2TokInx(sent, sentInx):
    blocks = [tokenizer.encode(i, add_special_tokens=False) for i in sent.split()]
    start_inx = len([j for ix,i in enumerate(blocks) for j in i if ix < sentInx])
    tok_inx = list(range(start_inx + 1, start_inx+len(blocks[sentInx])+1 )) # +1 is because bos token
    assert [tokenizer.encode(sent)[i] for i in tok_inx] == tokenizer.encode(sent.split()[sentInx], add_special_tokens=False), (sent, sentInx)
    return tok_inx

def extract_sentence(txt_path, gold_path, mark, prev=0,pos=None):
    df = pd.read_csv(txt_path, header=None, delimiter="\t")
    df.columns = ["target", "pos", "index", "sent1", "sent2"]
    gt = [int(i.strip()=="T") for i in open(gold_path, "r").readlines()]
    df['GT'] = gt
    if not pos is None:
        df = df[df["pos"] == pos]
    GT_list = df['GT'].tolist()
    if mark == "base":
        sents_list_1 = df['sent1'].tolist()
        sents_list_2 = df['sent2'].tolist()
        inx_list = df['index'].tolist()
    elif mark == "prompt1":
        target_list = df['target']
        sents_list_1 = ["The \"%s\" in this sentence \"%s\" means in one word : \"" % (i,j) for (i,j) in zip(target_list, df['sent1'].tolist())]
        sents_list_2 = ["The \"%s\" in this sentence:\"%s\" means in one word : \"" % (i,j) for (i,j) in zip(target_list, df['sent2'].tolist())]
        inx_list = ["%d-%d"%(len(i.split())-1, len(j.split())-1) for i,j in zip(sents_list_1, sents_list_2) ]
    elif mark == "prompt2":
        target_list = df['target']
        sents_list_1 = ["In this sentence \"%s\", \"%s\" means in one word : \"" % (j,i) for (i,j) in zip(target_list, df['sent1'].tolist())]
        sents_list_2 = ["In this sentence:\"%s\", \"%s\" means in one word : \"" % (j,i) for (i,j) in zip(target_list, df['sent2'].tolist())]
        inx_list = ["%d-%d"%(len(i.split())-1, len(j.split())-1) for i,j in zip(sents_list_1, sents_list_2) ]
    elif mark == "repeat": # TEST later
        sents_list_1 = ["%s %s" % (i,i) for i in df['sent1'].tolist()]
        sents_list_2 = ["%s %s" % (i,i) for i in df['sent2'].tolist()]
        inx_list = ["%d-%d"%(len(i.split())+int(inx.split("-")[0])-prev, len(j.split())+int(inx.split("-")[1])-prev) for i,j,inx in zip(df['sent1'], df['sent2'], df['index']) ]
    elif mark == "repeat_hint":
        sents_list_1 = ["Rewrite the sentence: %s , rewritten sentence: %s" % (i,i) for i in df['sent1'].tolist()]
        sents_list_2 = ["Rewrite the sentence: %s , rewritten sentence: %s" % (i,i) for i in df['sent2'].tolist()]
        inx_list = ["%d-%d"%(len(i.split())+int(inx.split("-")[0])-prev+6, len(j.split())+int(inx.split("-")[1])-prev+6) for i,j,inx in zip(df['sent1'], df['sent2'], df['index']) ]
    return sents_list_1, sents_list_2, GT_list, inx_list

def get_inx_raw(csv_path):
    df = pd.read_csv(csv_path)
    sents_list_1 = [i[:-1]+" "+i[-1] for i in df['sentence1']]
    sents_list_2 = [i[:-1]+" "+i[-1] for i in df['sentence2']]
    inx_1 = [s.split().index(w) if w in s.split() else -1 for w,s in zip(df['word'], sents_list_1)]
    inx_2 = [s.split().index(w) if w in s.split() else -1 for w,s in zip(df['word'], sents_list_2)]
    inx = [str(i1)+"-"+str(i2) for i1,i2 in zip(inx_1,inx_2)]
    df.insert(0, "inx", inx)
    df.to_csv("/home/liuzhu/LLM_LexSem/data/RAW/raw-c_inx.csv", index=False)


def extract_sentence_raw(csv_path, mark):
    wnl = WordNetLemmatizer()
    df = pd.read_csv(csv_path)
    sents_list_1 = [i[:-1]+" "+i[-1] for i in df['sentence1']]
    sents_list_2 = [i[:-1]+" "+i[-1] for i in df['sentence2']]
    GT_list = df['mean_relatedness'].tolist()
    # inx1_list = [[wnl.lemmatize(i) for i in s.split()].index(w) for w,s in zip(df['word'], sents_list_1)]
    # inx2_list = [[wnl.lemmatize(i) for i in s.split()].index(w) for w,s in zip(df['word'], sents_list_2)]
    # inx_list = [str(i)+"-"+str(j) for i,j in zip(inx1_list, inx2_list)]
    inx_list = df['inx'].tolist()

    if mark == "prompt1":
        sents_list_1 = ["The \"%s\" in this sentence: \"%s\" means in one word : \"" % (i,j) for (i,j) in zip(df['word'], sents_list_1)]
        sents_list_2 = ["The \"%s\" in this sentence: \"%s\" means in one word : \"" % (i,j) for (i,j) in zip(df['word'], sents_list_2)]
        inx_list = ["%d-%d"%(len(i.split())-1, len(j.split())-1) for i,j in zip(sents_list_1, sents_list_2) ]

    elif mark == "prompt2":
        sents_list_1 = ["In this sentence \"%s\", \"%s\" means in one word : \"" % (j,i) for (i,j) in zip(df['word'], sents_list_1)]
        sents_list_2 = ["In this sentence:\"%s\", \"%s\" means in one word : \"" % (j,i) for (i,j) in zip(df['word'], sents_list_2)]
        inx_list = ["%d-%d"%(len(i.split())-1, len(j.split())-1) for i,j in zip(sents_list_1, sents_list_2) ]

    elif mark == "repeat":
        inx_list = [str(len(s1.split())+int(ix.split("-")[0]))+"-"+str(len(s2.split())+int(ix.split("-")[1])) for s1,s2,ix in zip(sents_list_1,sents_list_2,inx_list)]
        sents_list_1 = ["%s %s" % (i,i) for i in sents_list_1]
        sents_list_2 = ["%s %s" % (i,i) for i in sents_list_2]

    elif mark == "repeat_prev":
        inx_list = [str(len(s1.split())+int(ix.split("-")[0])-1)+"-"+str(len(s2.split())+int(ix.split("-")[1])-1) for s1,s2,ix in zip(sents_list_1,sents_list_2,inx_list)]
        sents_list_1 = ["%s %s" % (i,i) for i in sents_list_1]
        sents_list_2 = ["%s %s" % (i,i) for i in sents_list_2]

    return sents_list_1, sents_list_2, GT_list, inx_list

def get_batch_data(sents_list, inx_list, batch_size=50, prompt=None):
    batch_list = []
    tar_inx_list = []
    for i in range(len(sents_list)//batch_size+1):
        if i*batch_size == len(sents_list):
            break
        sent = sents_list[i*batch_size:min(len(sents_list),(i+1)*batch_size)]
        inx = inx_list[i*batch_size:min(len(sents_list),(i+1)*batch_size)]
        tar_inx = [SentInx2TokInx(s,ix) for s,ix in zip(sent, inx)]
        encoded_sentence = tokenizer(sent, return_tensors="pt", padding=True)
        batch_list.append(encoded_sentence)
        tar_inx_list.append(tar_inx)
    return batch_list, tar_inx_list

def get_layer_representations(model, encoded_sentences, tar_inx):
    encoded_sentences.to(device)
    model.eval()
    with torch.no_grad():
        output = model(**encoded_sentences, output_hidden_states=True)
    all_layer_representations = output.hidden_states # the first is embedding layer
    
    represents = torch.zeros(all_layer_representations[0].shape[0], len(all_layer_representations), all_layer_representations[0].shape[2])
    for ix,i in enumerate(all_layer_representations):
        for j in range(i.shape[0]):
            represents[j, ix] = i[j, tar_inx[j]].mean(dim=0)

    return represents

def evaluation(pred, GT, type="acc"):
    assert len(pred) == len(GT), print(len(pred), len(GT))
    if type == "acc":
        result = sum([i==j for i,j in zip(pred, GT)]) / len(pred)
    elif type == "corr":
        result, _ = spearmanr(pred, GT)

    return result

def anisotropy_removal(embedding):
    return (embedding - embedding.mean(dim=0)) / (embedding.std(dim=0) + 1e-6)

#%%
if __name__ == "__main__":
    root = "/home/liuzhu/"#"YOUR_ROOT_DIR_PATH"
    mode = "TEST" # or TEST arg
    aniso = True # arg
    if mode == "EVAL":
        txt_path = root+"LLM_LexSem/data/dev/dev.data.txt"
        gold_path = root+"LLM_LexSem/data/dev/dev.gold.txt"
    else:
        txt_path = root+"LLM_LexSem/data/test/test.data.txt"
        gold_path = root+"LLM_LexSem/data/test/test.gold.txt"
        acc_save_path = root + "LLM_LexSem/output/test_acc.csv" if aniso else \
                        root + "LLM_LexSem/output/test_acc_no.csv" 
    threshold_path = root+"LLM_LexSem/thresholds.csv" if aniso else \
                     root+"LLM_LexSem/thresholds_no.csv"
    if os.path.exists(threshold_path):
        thresholds = pd.read_csv(threshold_path, delimiter=" ") # obtained by a validate set
    else:
        thresholds = pd.DataFrame()
    raw_path = root+"LLM_LexSem/data/RAW/raw-c_inx.csv"
    mark = "repeat" # args [repeat, repeat_prev, prompt1, prompt2]
    dataset = "WiC" # args
    pos = None # args
    tokenizer = LlamaTokenizer.from_pretrained("/data61/liuzhu/LLM/llama-main/llama-2-7b-hf")
    tokenizer.pad_token = '[PAD]'

    if dataset == "WiC":
        sents_list_1, sents_list_2, GT_list, inx_list = extract_sentence(txt_path, gold_path, mark, prev=0, pos=pos)
    else:
        sents_list_1, sents_list_2, GT_list, inx_list = extract_sentence_raw(raw_path, mark)
    print(sents_list_1[0], sents_list_2[0], GT_list[0], inx_list[0])
    inx_list_1 = [int(i.split("-")[0]) for i in inx_list]
    inx_list_2 = [int(i.split("-")[1]) for i in inx_list]
    batch_list_1, tar_inx_list_1 = get_batch_data(sents_list_1, inx_list_1, batch_size=30)
    batch_list_2, tar_inx_list_2 = get_batch_data(sents_list_2, inx_list_2, batch_size=30)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LlamaForCausalLM.from_pretrained("/data61/liuzhu/LLM/llama-main/llama-2-7b-hf").half().to(device)

    data_pred_layers = {}
    data_sim_layers = {}
    for threshold in tqdm(range(0,100,5)):
        for batch_1, batch_2, tar_inx_1, tar_inx_2 in tqdm(zip(batch_list_1, batch_list_2, tar_inx_list_1, tar_inx_list_2),total=len(batch_list_1)):
            batch_rep_1 = get_layer_representations(model, batch_1.to(device), tar_inx_1)
            batch_rep_2 = get_layer_representations(model, batch_2.to(device), tar_inx_2)

            for i in range(33):
                if aniso:
                    batch_sim = torch.nn.functional.cosine_similarity( anisotropy_removal(batch_rep_1[:,i]), anisotropy_removal(batch_rep_2[:,i]) ).cpu()
                else:
                    batch_sim = torch.nn.functional.cosine_similarity( batch_rep_1[:,i], batch_rep_2[:,i] ).cpu()
                if mode == "TEST":
                    batch_pred = (batch_sim > thresholds.at[i, mark]).int().tolist()
                    if i in data_pred_layers:
                        data_pred_layers[i] += batch_pred
                        data_sim_layers[i] += batch_sim.tolist()
                    else:
                        data_pred_layers[i] = batch_pred
                        data_sim_layers[i] = batch_sim.tolist()
                else:
                    batch_pred = (batch_sim > threshold/100).int().tolist()
                    if threshold not in data_pred_layers:
                        data_pred_layers[threshold] = {}
                    if i not in data_pred_layers[threshold]:
                        data_pred_layers[threshold][i] = []
                    data_pred_layers[threshold][i] += batch_pred
        if mode == "TEST":
            break

    if mode == "TEST":
        if dataset == "WiC":
            acc_layers = [evaluation(pred, GT_list) for _, pred in data_pred_layers.items()]
            print(acc_layers)
            print("The best layer: %d with acc: %.3f" % (np.argmax(acc_layers), np.max(acc_layers)))
            plt.title(mark)#'repeat_prev')#)
            plt.plot(acc_layers)
            if os.path.exists(acc_save_path):
                df_acc = pd.read_csv(acc_save_path)
            else:
                df_acc = pd.DataFrame()
                df_acc["Index"] = range(len(acc_layers))
            df_acc[mark] = acc_layers#mark
            df_acc.to_csv(acc_save_path, index=False)

        else: # RAW
            corr_layers = [evaluation(sim, GT_list, type="corr") for _, sim in data_sim_layers.items()]
            print(np.argmax([round(i*100,1) for i in corr_layers]))
            print([round(i*100,1) for i in corr_layers])
            print(corr_layers)
            plt.plot(corr_layers)
    else:
        acc_layers = [[evaluation(pred, GT_list) for _, pred in preds.items()] for _, preds in data_pred_layers.items()]
        acc_layers = np.array(acc_layers)
        best_th_layers = [list(range(0,100,5))[i]/100 for i in np.argmax(acc_layers, axis=0)]
        best_acc_layers = np.max(acc_layers, axis=0)
        plt.plot(best_acc_layers)
        plt.plot(best_th_layers)
        for ix,(th,acc) in enumerate(zip(best_th_layers, best_acc_layers)):
            print("The best accuracy: %.2f\tthreshold: %.2f\tlayer: %d" % (acc,th,ix))
        thresholds['Index'] = range(len(best_th_layers))
        thresholds[mark] = best_th_layers
        thresholds.to_csv(root+"LLM_LexSem/thresholds.csv", sep=" ", index=False)

# %%
