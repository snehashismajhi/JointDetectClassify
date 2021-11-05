# Weakly-supervised Joint Anomaly Detection and Classification
Anomaly activities such as robbery, explosion, accidents, etc. need immediate actions for preventing loss of human life and property in real world surveillance systems. Although the recent automation in surveillance systems are capable of detecting the anomalies, but they still need human efforts for categorizing the anomalies and taking necessary preventive actions. This is due to the lack of methodology performing both anomaly detection and classification for real world scenarios. Thinking of a fully automatized surveillance system, which is capable of both detecting and classifying the anomalies that need immediate actions, a joint anomaly detection and classification method is a pressing need. The task of joint detection and classification of anomalies becomes challenging due to the unavailability of dense annotated videos pertaining to anomalous classes, which is a crucial factor for training modern deep architecture. Furthermore, doing it through manual human effort seems impossible. Thus, we propose a method that jointly handles the anomaly detection and classification in a single framework by adopting a weakly-supervised learning paradigm. In weakly-supervised learning instead of dense temporal annotations, only video-level labels are sufficient for learning. The proposed model is validated on a large-scale publicly available UCF-Crime dataset, achieving state-of-the-art results.

# Proposed Framework
![Prposed Framework](https://github.com/snehashismajhi/JointDetectClassify/blob/main/FG21%20Framework.jpg)

# First-Level Attention Module
![First-Level Attention Module](https://github.com/snehashismajhi/JointDetectClassify/blob/main/First-Level%20Attention.jpg)

# Second-Level Attention Module
![Second-Level Attention Module](https://github.com/snehashismajhi/JointDetectClassify/blob/main/Second-Level%20Attention.jpg)

# Anomaly Detection Performance Comparision
![Anomaly Detection Performance Comparision](https://github.com/snehashismajhi/JointDetectClassify/blob/main/AUC%20Compare.JPG)

# Anomaly Classification Performance
![Anomaly Classification Performance](https://github.com/snehashismajhi/JointDetectClassify/blob/main/Confusion%20Matrix.jpg)
