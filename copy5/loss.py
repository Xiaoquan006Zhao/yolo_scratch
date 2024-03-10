import torch
import torch.nn as nn 
from utils import (
	ciou,
	convert_cells_to_bboxes
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

		no_object_loss = self.bce(pred[..., 0:1][no_obj], target[..., 0:1][no_obj])
		object_loss = self.bce(pred[..., 0:1][obj], target[..., 0:1][obj])
	
		box_preds = torch.cat([self.sigmoid(pred[..., 1:3]), torch.exp(pred[..., 3:5]) * anchors],dim=-1)

		ious = 1-ciou(box_preds[obj], target[..., 1:5][obj])
		box_loss = ious.mean()

		class_loss = self.cross_entropy((pred[..., 5:][obj]), target[..., 5][obj].long()) 

		return ( 
			box_loss 
			+ object_loss 
			+ no_object_loss 
			+ class_loss 
		)