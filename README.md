
#  **Graph-Anomaly-Detection**

- The database integrates various elements related to graph anomaly detection algorithms, including papers, datasets, code, and other relevant materials.



## **图异常检测研究的时间线**

[![timeline](Timeline.png)](https://ieeexplore.ieee.org/abstract/document/9565320)



## **Surveys**

| 序号 |                           论文名称                           | 年份 |
| :--: | :----------------------------------------------------------: | :--: |
|  1   | [Combining machine learning with knowledge engineering to detect fake news in social networks-a survey](https://arxiv.org/abs/2201.08032) | 2022 |
|  2   | [A Comprehensive Survey on Graph Anomaly Detection With Deep Learning](https://ieeexplore.ieee.org/abstract/document/9565320) | 2021 |
|  3   | [Deep Learning for Anomaly Detection: A Review](https://dl.acm.org/doi/abs/10.1145/3439950) | 2021 |
|  4   | [Anomaly detection for big datausing efficient techniques: A review](https://link.springer.com/chapter/10.1007/978-981-15-3514-7_79) | 2021 |
|  5   | [Fraud detection: A systematic literature review of graph-based anomaly detectionapproaches](https://www.sciencedirect.com/science/article/pii/S0167923620300580) | 2020 |
|  6   | [A comprehensive surveyof anomaly detection techniques for high dimensional big data](https://journalofbigdata.springeropen.com/articles/10.1186/s40537-020-00320-x) | 2020 |
|  7   | [Outlier detection: Methods,models, and classification](https://dl.acm.org/doi/10.1145/3381028) | 2020 |
|  8   | [Anomalous instance detection in deep learning: A survey](https://www.semanticscholar.org/paper/Anomalous-Instance-Detection-in-Deep-Learning:-A-Bulusu-Kailkhura/86c12bb6fb6fb2956d1147bd1b15a788b6d07f6e) | 2020 |
|  9   | [Machine learning techniques for network anomaly detection: A survey](https://ieeexplore.ieee.org/abstract/document/9089465) | 2020 |
|  10  | [Deep learning for anomaly detection: A survey](https://arxiv.org/abs/1901.03407) | 2019 |
|  11  | [A comprehensive survey on network anomaly detection](https://link.springer.com/article/10.1007/s11235-018-0475-8) | 2019 |
|  12  | [A survey of deep learning-based network anomaly detection](https://link.springer.com/article/10.1007/s10586-017-1117-8) | 2019 |
|  13  | [A survey on social media anomaly detection](https://dl.acm.org/doi/10.1145/2980765.2980767) | 2016 |
|  14  | [Graph based anomaly detectionand description: A survey](https://link.springer.com/article/10.1007/s10618-014-0365-y) | 2015 |
|  15  | [Anomaly detection in dynamic networks: A survey](https://wires.onlinelibrary.wiley.com/doi/10.1002/wics.1347) | 2015 |
|  16  | [Anomaly detection in online social networks](https://www.sciencedirect.com/science/article/pii/S0378873314000331) | 2014 |
|  17  | [A survey ofoutlier detection methods in network anomaly identification](https://ieeexplore.ieee.org/abstract/document/8130440) | 2011 |
|  18  | [Anomaly detection: A survey](https://dl.acm.org/doi/10.1145/1541880.1541882) | 2009 |

## **ANOMALOUS NODE DETECTION**

![image-20240110153655311](C:\Users\Le'novo\AppData\Roaming\Typora\typora-user-images\image-20240110153655311.png)

三类异常节点：结构异常、社区异常和全局异常。

- 结构异常：仅考虑图的结构信息
- 社区异常：同时考虑节点特征和图的结构信息
- 全局异常：仅考虑节点的特征

| 序号 |                           文章名称                           | 年份 |    模型    |                             代码                             |
| :--: | :----------------------------------------------------------: | :--: | :--------: | :----------------------------------------------------------: |
|      | [Selective network discovery via deep reinforcement learning on embedded spaces](https://appliednetsci.springeropen.com/articles/10.1007/s41109-021-00365-8) | 2021 |    NAC     |                             ---                              |
|      | [Decoupling representation learning and classification for gnn-based anomaly detection](https://dl.acm.org/doi/10.1145/3404835.3462944) | 2021 |    DCI     |         [code](https://github.com/wyl7/DCI-pytorch)          |
|      | [Resgcn:Attention-based deep residual modeling for anomaly detection on attributed networks](https://dl.acm.org/doi/abs/10.1007/s10994-021-06044-0) | 2021 |   ResGCN   |         [code](https://bitbucket.org/paulpei/resgcn)         |
|      | [Outlier resistant unsupervised deep architectures for attributed network embedding](https://dl.acm.org/doi/10.1145/3336191.3371788) | 2020 |    DONE    |        [code](https://github.com/vasco95/DONE_AdONE)         |
|      | [A deep multi-view framework for anomaly detection on attributed networks](https://ieeexplore.ieee.org/abstract/document/9162509/) | 2020 |   ALARM    |                             ---                              |
|      | [Gcn-based user representation learning for unifying robust recommendation and fraudster detection](https://dl.acm.org/doi/abs/10.1145/3397271.3401165) | 2020 |  GraphRfi  |         [code](https://github.com/zsjdddhr/GraphRfi)         |
|      | [Anomalydae: Dual autoencoder foranomaly detection on attributed networks](https://ieeexplore.ieee.org/document/9053387/) | 2020 | AnomalyDAE |        [code](https://github.com/haoyfan/AnomalyDAE)         |
|      | [Inductive anomaly detection on attributed networks](https://dl.acm.org/doi/10.5555/3491440.3491619) | 2020 |   AEGIS    |                             ---                              |
|      | [Deep anomaly detection on attributed networks](https://epubs.siam.org/doi/abs/10.1137/1.9781611975673.67) | 2019 |  DOMINANT  | [code](https://github.com/kaize0409/GCN_AnomalyDetection_pytorch) |
|      | [A semi-supervised graph attentive network for financial fraud detection](https://ieeexplore.ieee.org/document/8970829) | 2019 |  SemiGNN   | [code](https://github.com/safe-graph/DGFraud/tree/master/algorithms/SemiGNN) |
|      | [Interactive anomaly detection on attributednetworks](https://dl.acm.org/doi/10.1145/3289600.3290964) | 2019 |  GraphUCB  |                             ---                              |
|      | [Specae: Spectral auto encoder for anomaly detection in attributed networks](https://dl.acm.org/doi/10.1145/3357384.3358074) | 2019 |   SpecAE   |                             ---                              |
|      | [Fdgars: Fraudster detection via graph convolutional networks in online app review system  ](https://dl.acm.org/doi/10.1145/3308560.3316586) | 2019 |   Fdgars   |                             ---                              |
|      | [Anomalous: A joint modeling approach for anomaly detection on attributed networks](https://www.ijcai.org/Proceedings/2018/0488.pdf) | 2018 | Anomalous  |         [code](https://github.com/zpeng27/ANOMALOUS)         |
|      | [Adaptive spammer detection with sparse group modeling](https://aaai.org/papers/00319-14887-adaptive-spammer-detection-with-sparse-group-modeling/) | 2017 |   SGASD    |                             ---                              |
|      | [Accelerated local anomaly detection via resolving attributed networks](https://www.ijcai.org/Proceedings/2017/0325.pdf) | 2017 |    ALAD    |         [code](https://github.com/ninghaohello/ALAD)         |
|      | [Radar: Residual analysis for anomaly detection in attributed networks](https://dl.acm.org/doi/10.5555/3172077.3172187) | 2017 |   Radar    |       [code](https://github.com/szumbrunn/radar-java)        |
|      | [An embedding approach to anomaly detection](https://ieeexplore.ieee.org/document/7498256) | 2016 |    ---     |                             ---                              |
|      |                                                              |      |            |                                                              |
|      |                                                              |      |            |                                                              |

