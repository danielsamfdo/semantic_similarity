import glob
import re
def collecting_data(filesregexmatch, maxsplit):
  Data = []
  for fname in glob.glob(filesregexmatch):
    with open(fname) as f:
      lines = f.readlines()
      # print fname
      with open(re.sub("input","gs",fname)) as nfile:
        # print fname
        nlines = nfile.readlines()
        for index,line in enumerate(lines):
          if len(line.split('\t')) != maxsplit :
            print "THERE IS AN INCONSISTENCY PLEASE CHECK THE BELOW FILE ....."
            print fname
            break;
          else:
            # Does not include the ones which do not have numbers
            if( not re.search("(\d)((\.)(\d))?", nlines[index])):
                continue;
            x = line.split('\t')
            Data.append([x[0],x[1], float(re.sub("\n","",nlines[index]))])    
  return Data

def TrainData():
  prefix = "/Users/danielsampetethiyagu/github/semantic_similarity/STS-data"
  Data = collecting_data(prefix + "/*201[1-5]*/*input*.txt", 2) + collecting_data(prefix + "/*2016*/*input*.txt", 4) 
  return Data