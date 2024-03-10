import torch
import torch.nn as nn 
from utils import (
	ciou,
)

class YOLOLoss(nn.Module): 
	def __init__(self): 
		super().__init__() 
		self.mse = nn.MSELoss() 
		self.bce = nn.BCEWithLogitsLoss() 
		self.cross_entropy = nn.CrossEntropyLoss() 
		self.sigmoid = nn.Sigmoid() 
	
	def forward(self, pred, target, anchors): 
		anchors = anchors.reshape(1, 3, 1, 1, 2) 

		obj = target[..., 0] == 1
		no_obj = target[..., 0] == 0

		no_object_loss = self.bce((pred[..., 0:1][no_obj]), (target[..., 0:1][no_obj]), ) 

		box_preds = torch.cat([self.sigmoid(pred[..., 1:3]), torch.exp(pred[..., 3:5]) * anchors],dim=-1)
		ious = ciou(box_preds[obj], target[..., 1:5][obj])
		box_loss = (1-ious).mean()

		# The way I understand why ious*target is that since the objectiveness and bbox is
		# tied together. If the bbox has small or no overlap, we should discard it's objectiveness
		# prediction as well and penalize the model more for confident objectiveness prediction
		# that has no bbox to back the confidence up
		object_loss = self.mse(self.sigmoid(pred[..., 0:1][obj]), ious * target[..., 0:1][obj]) 

		class_loss = self.cross_entropy((pred[..., 5:][obj]), target[..., 5][obj].long()) 

		return ( 
			box_loss 
			+ object_loss 
			+ no_object_loss 
			+ class_loss 
		)
		

