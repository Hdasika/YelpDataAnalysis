# word cloud sample with any shap
# need further edits , I keep pushing the code


from wordcloud import WordCloud, STOPWORDS # python -m pip install wordcloud
from PIL import Image
import urllib
import requests
import numpy as np
import matplotlib.pyplot as plt
# this is the word cloud sample text 


# the text upon which the word cloud is generated 
words = 'access guest guest apartment area area bathroom bed bed bed bed bed bedroom block coffee coffee coffee coffee entrance entry francisco free garden guest home house kettle kettle kitchen kitchen kitchen kitchen kitchen kitchenliving located microwave neighborhood new park parking place privacy private queen room san separate seperate shared space space space street suite time welcome It has previously been suggested that neurons with line and edge selectivities found in primary visual cortex of cats and monkeys form a sparse, distributed representation of natural scenes, and it has been reasoned that such responses should emerge from an unsupervised learning algorithm that attempts to find a factorial code of independent visual features. We show here that a new unsupervised learning algorithm based on information maximization, a nonlinear "infomax" network, when applied to an ensemble of natural scenes produces sets of visual filters that are localized and oriented. Some of these filters are Gabor-like and resemble those produced by the sparseness-maximization network. In addition, the outputs of these filters are as independent as possible, since this infomax network performs Independent Components Analysis or ICA, for sparse (super-gaussian) component distributions. We compare the resulting ICA filters and their associated basis functions, with other decorrelating filters produced by Principal Components Analysis (PCA) and zero-phase whitening filters (ZCA). The ICA filters have more sparsely distributed (kurtotic) outputs on natural scenes. They also resemble the receptive fields of simple cells in visual cortex, which suggests that these neurons form a natural, information-theoretic coordinate system for natural images.'
mask = np.array(Image.open(requests.get('https://www.thewrap.com/wp-content/uploads/2018/09/the-nun-conjuring-universe-post-credits-scene.jpg', stream=True).raw))

# This function generating wordcloud. 
def generate_wordcloud(words, mask):
    word_cloud = WordCloud(width = 512, height = 512, background_color='white', stopwords=STOPWORDS, mask=mask).generate(words)
    plt.figure(figsize=(10,8),facecolor = 'white', edgecolor='blue')
    plt.imshow(word_cloud)
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.show()
# Generate your wordcloud
generate_wordcloud(words, mask)