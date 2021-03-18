from pytorch_metric_learning.losses import TripletMarginLoss,ContrastiveLoss,ArcFaceLoss
from pytorch_metric_learning.distances import LpDistance,SNRDistance,CosineSimilarity
'''
#TripletMarginLoss with unnormalized L1 distance
distance = LpDistance(normalize_embeddings=False, p=1)
loss_func = TripletMarginLoss(distance=distance)
'''

'''
#TripletMarginLoss with signal-to-noise ratio
distance = SNRDistance()
loss_func = TripletMarginLoss(distance=distance)
'''

'''
#TripletMarginLoss with cosine similarity
distance = CosineSimilarity()
loss_func = TripletMarginLoss(distance=distance)
'''

'''
from pytorch_metric_learning.reducers import ThresholdReducer
from pytorch_metric_learning.losses import MultiSimilarityLoss

reducer = ThresholdReducer(low=10, high=30)
loss_func = MultiSimilarityLoss(reducer=reducer)
'''

'''
from pytorch_metric_learning.regularizers import LpRegularizer
loss_func = ContrastiveLoss(embedding_regularizer=LpRegularizer())
'''

'''
from pytorch_metric_learning.regularizers import RegularFaceRegularizer
loss_func = ArcFaceLoss(weight_regularizer=RegularFaceRegularizer())
'''