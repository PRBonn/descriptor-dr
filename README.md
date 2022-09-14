# descriptor-dr

## Supplementary

### Embedding visualization of the descriptors

<img src="pics/SIFT.svg" width="230"><img src="pics/SIFT-PCA-64.svg" width="230"><img src="pics/SIFT-Ours-SV-64.svg" width="230">  <br />
<img src="pics/MKD.svg" width="230"><img src="pics/MKD-PCA-64.svg" width="230"><img src="pics/MKD-Ours-SV-64.svg" width="230">  <br />
<img src="pics/TFeat.svg" width="230"><img src="pics/TFeat-PCA-64.svg" width="230"><img src="pics/TFeat-Ours-SV-64.svg" width="230">  <br />
<img src="pics/HardNet.svg" width="230"><img src="pics/HardNet-PCA-64.svg" width="230"><img src="pics/HardNet-Ours-SV-64.svg" width="230">  <br />




We provide the t-SNE embedding visualization of the descriptors on UBC Phototour dataset Liberty sequence. 
We visualize the embeddings of SIFT, MKD, TFeat, and HardNet generated using PCA and supervised methods by mapping high-dimensional descriptors (128 and 64) into 2D using t-SNE visualization. We pass the image patches through the descriptor extractor, followed by PCA or MLPs, to get lower-dimensional descriptors and determine their 2D locations using t-SNE transformation. Finally, we visualize the entire patch at each location.

From the visualization, we can observe similar results as we discussed in the paper. For SIFT and MKD, the original descriptor space is irregular, and similar and dissimilar features are overlapped. Therefore, PCA projection will keep this irregular structure of the descriptor space. However, after learning a more discriminative representation using triplet loss, similar image patches in the descriptor space are close to each other while dissimilar ones have distances from each other. For TFeat and HardNet, since the outputting space is already optimized for the $l_2$ metric, image patches in the descriptor space are already separated well based on their appearance. Therefore, a simple PCA can preserve this distinctive structure and perform on par compared with the learned projection. 


### More experienments on patch pair verification, image matching, and patch retrieval
(TBD)

