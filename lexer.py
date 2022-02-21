import re

toks = { "split":"</s>"
       , "unk":"<unk>"
       , "pad":"<pad>"
       , "eot":"<eot>"
       }

def create_tokens(txt,avoid_tokens=[]):
    # create_tokens splits input into lowercase words
    special_tokens = [toks[k] for k in toks]
    rx = re.compile("[a-zA-Z0-9æøåÆØÅ']+(?=\s|,|.|\(|\)|\?|\;|\:|<|>|\!|-|_)|" + "|".join(special_tokens))

    txt = txt.lower()
    txt = txt.replace("&","and")
    #dk bogstaver
    txt = txt.replace("aa","å")
    #txt = txt.replace("Aa","å")
    txt = txt.replace("é","e")
    txt = txt.replace("è","e")
    txt = txt.replace("ë","e")
    txt = txt.replace("ē","e")
    txt = txt.replace("ə","e")
    

    txt = txt.replace("â","a")
    txt = txt.replace("ä","a")

    txt = txt.replace("ö","o")
    txt = txt.replace("ŏ","o")
    txt = txt.replace("ó","o")
    txt = txt.replace("ō","o")
    txt = txt.replace("ü","u")

    txt = txt.replace("í","i")

    txt = txt.replace("á","a")
    txt = txt.replace("à","a")
    txt = txt.replace("ā","a")
    txt = txt.replace("ǣ","æ")
    txt = txt.replace("ɐ","a")

    txt = txt.replace("ß","ss")

    txt = txt.replace("ñ","n")
    txt = txt.replace("ħ","n")

    txt = txt.replace("ð","d")

    txt = txt.replace("”"," ")
    
    txt = txt.replace(u"\u00A0"," ")
    txt = txt.replace("\t"," ")

    ts = rx.findall(txt + " ")

    ts = [x for x in ts if x != "'"]

    if avoid_tokens != None and len(avoid_tokens) > 0:
        res = [x for x in ts if x not in avoid_tokens]
    else:
        res = ts

    return res

def remove_comments(txt):
    rx = re.compile("//[^\n]*\n")
    txt_res = rx.sub("...\n",txt)
    return txt_res
