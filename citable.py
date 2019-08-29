import os 
import requests
from zipfile import ZipFile
import shutil

class Citable:

    def __init__(self, DOI):
        
        self.doi = DOI.split('/')[-1].split('.')[1]
        self.url = 'https://zenodo.org/api/records/' + self.doi
        
        if not os.path.isdir('./data'):
            os.mkdir('./data')
        self.download()
    
    def download(self):
        r = requests.get(self.url)
        linksfiles = [i['links']['self'] for i in r.json()['files'] if not (os.path.isfile('./data/' + i['key']) or os.path.isfile('./data/' + '.'.join(i['key'].split('.')[:-1])))]
        if not linksfiles:
            print('All files of the repository are already in the local data folder.')
        else:
            # going through all links, download and extract all files:
            for URL in linksfiles:
                try:
                    dl = requests.get(URL)

                    # write file into the 'data'-folder and name it test.zip:
                    if URL.split('.')[-1] == 'zip':
                        with open('./data/test.zip', 'wb') as f:  
                            f.write(dl.content)

                        # extract zip-file into data-folder
                        zf = ZipFile('./data/test.zip', 'r')
                        zf.extractall('./data')
                        zf.close()

                        # removing original zip-file and '__MACOSX'-folder, if it exists:
                        os.remove('./data/test.zip') 
                        if os.path.isdir('./data/__MACOSX'):
                            shutil.rmtree('./data/__MACOSX')
                    else:
                        with open('./data/' + URL.split('/')[-1], 'wb') as f:  
                            f.write(dl.content)
                    print('Successfully downloaded and extracted data under ' + URL)
                except:
                    print('Something went wrong while downloading and extracting data under ' + URL)
