import torchvision.models as models
from efficientnet_pytorch import EfficientNet
import torch.nn as nn
import torch
import os

import numpy as np
import torch


class BaseModel():
    def name(self):
        return 'base_model'

    def __init__(self, config):
        self.config = config
        self.use_weighted_loss = config["use_weighted_loss"]
        self.continue_train = config["continue_train"]
        self.freeze = True if "freeze" in config and config["freeze"]!=-1 else False
        self.trainable_layer_num = config["freeze"] if self.freeze else None

    def forward(self):
        pass

    def optimize_parameters(self):
        pass

    def get_current_errors(self):
        pass

    def save_state(self):
        pass

    def load_state(self):
        pass

class DeepModel(BaseModel):
    def name(self):
        n = [self.deep_model]
        if self.use_pretrained:
            n += ['pretrained_weights']
        if self.use_weighted_loss:
            n += ['weighted_loss']
        if self.use_antialias:
            n += ['antialias']

        return '_'.join(n)

    def __init__(self, config, is_eval=False, class_weight=None, device=None):
        """
        TODO: very messy. Should clean this.
        TODO: is_eval param is not being used. Should use
        TODO: NNs should use 1 output neuron for binary classification instead of 2. Should refactor to use 1 output with last linear layer 1 output then sigmoid
        https://stats.stackexchange.com/questions/207049/neural-network-for-binary-classification-use-1-or-2-output-neurons
        """
        super().__init__(config)
        print("Config File:")
        print(config)
        self.is_eval = is_eval
        self.deep_model = self.config["deep_model"]
        self.class_weight = class_weight if self.use_weighted_loss else None
        self.MixUp = True if 'mix_up' in self.config and self.config['mix_up']['mix_up'] else False

        # using efficientnet
        if ("efficientnet") in self.deep_model:
            model = EfficientNet.from_pretrained(self.deep_model, num_classes=self.config["num_subtypes"])

        else :
            model = getattr(models, self.deep_model)
            model = model(**self.config["parameters"])
            if "feature_extract" in self.config and self.config["feature_extract"]:
                for param in model.parameters():
                    param.requires_grad = False

            # Modified the original HACK! part into an if condition to avoid the need to comment/uncomment lines
            # when using vgg vs resnet/inception
            if 'vgg' in self.deep_model:
                model.classifier._modules['6'] = torch.nn.Linear(4096, self.config["num_subtypes"])
            elif 'mobilenet' in self.deep_model:
                model.classifier[1] = torch.nn.Linear(in_features=model.classifier[1].in_features, out_features=self.config["num_subtypes"])
            else:
                num_features = model.fc.in_features
                model.fc = torch.nn.Linear(num_features, self.config["num_subtypes"])

        print(model)
        if device is not None:
            self.model = torch.nn.DataParallel(model, device_ids=range(0,len(device))).cuda()
            self.model = model.to(f'cuda:0')
        else:
            if torch.cuda.is_available():
                self.model = model.cuda()
            else:
                self.model = model

        if not self.is_eval:
            if self.use_weighted_loss and self.class_weight is not None:
                weight = torch.Tensor(self.class_weight).cuda()
                self.criterion = torch.nn.CrossEntropyLoss(
                    reduction='mean', weight=weight)
                #self.criterion = torch.nn.BCEWithLogitsLoss(
                #    reduction='mean', weight=torch.from_numpy(self.class_weight).cuda())
            else:
                self.criterion = torch.nn.CrossEntropyLoss(reduction='mean')
                #self.criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')

        print("Parameters to learn:")
        if self.freeze:
            for param in model.parameters():
                param.requires_grad = False
            rev_child = reversed(list(model.children()))
            self.freeze_layers(rev_child, self.trainable_layer_num)

            params_to_update = []
            for name,param in model.named_parameters():
                if param.requires_grad == True:
                    params_to_update.append(param)
                    print("\t", name)
        else:
            params_to_update = model.parameters()
            for name,param in model.named_parameters():
                if param.requires_grad == True:
                    print("\t", name)

        optimizer = getattr(torch.optim, self.config["optimizer"]["type"])
        self.optimizer = optimizer(params_to_update, **self.config["optimizer"]["parameters"])

        if "scheduler" in self.config:
            scheduler = getattr(torch.optim.lr_scheduler, self.config["scheduler"]["type"])
            self.scheduler = scheduler(self.optimizer, **self.config["scheduler"]["parameters"])

        if self.continue_train:
            self.load_state(config["load_deep_model_id"], device=device)

        if self.is_eval:
            self.load_state(config["load_deep_model_id"], device=device)
            self.model = self.model.eval()


    def forward(self, input_data):
        output = self.model.forward(input_data)

        if type(output).__name__ in ['GoogLeNetOutputs', 'InceptionOutputs'] and config["parameters"]["aux_logits"]:
            logits = output.logits
        else:
            logits = output

        probs = torch.softmax(logits, dim=1)
        return logits, probs, output

    def freeze_layers(self, rev_child, num):
        for layer in rev_child:
            if num > 0:
                if isinstance(layer, nn.Sequential) or \
                   isinstance(layer, models.resnet.BasicBlock) or \
                   isinstance(layer, models.resnet.Bottleneck) or \
                   isinstance(layer, models.squeezenet.Fire):
                    num = self.freeze_layers(reversed(list(layer.children())), num)
                else:
                    if sum(p.numel() for p in layer.parameters())!=0:
                        for param in layer.parameters():
                            param.requires_grad = True
                        num -= 1
        return num

    def get_loss(self, logits, labels, output=None,
                 labels_mixed=None, lam=None):
        if type(output).__name__ == 'GoogLeNetOutputs':
            loss = self.criterion(logits.type(torch.float), labels.type(torch.long)) + 0.4 * (self.criterion(output.aux_logits1.type(
                torch.float), labels.type(torch.long)) + self.criterion(output.aux_logits2.type(torch.float), labels.type(torch.long)))
        elif type(output).__name__ == 'InceptionOutputs':
            loss = self.criterion(logits.type(torch.float), labels.type(
                torch.long)) + 0.4 * self.criterion(output.aux_logits.type(torch.float), labels.type(torch.long))
        else:
            if self.MixUp and labels_mixed is not None and lam is not None:
                loss = lam * self.criterion(logits.type(
                    torch.float), labels.type(torch.long)) + (1 - lam) * self.criterion(logits.type(
                        torch.float), labels_mixed.type(torch.long))
            else:
                loss = self.criterion(logits.type(
                    torch.float), labels.type(torch.long))
        return loss

    def optimize_parameters(self, logits, labels, output=None,
                            labels_mixed=None, lam=None):
        self.loss = self.get_loss(logits, labels, output, labels_mixed, lam)
        self.loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return self.loss.item()

    def get_current_errors(self):
        return self.loss.item()

    def scheduler_step(self):
        self.scheduler.step()

    def get_current_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def load_state(self, save_path, device=None):
        if device:
            state = torch.load(save_path, map_location=device)
        elif not torch.cuda.is_available():
            state = torch.load(save_path, map_location='cpu')
        else:
            state = torch.load(save_path)

        try:
            self.model.load_state_dict(state['state_dict'])
        except RuntimeError:
            pretrained_dict = state['state_dict']
            model_dict = self.model.state_dict()
            # filter out unnecessary keys
            pretrained_dict = {k: v for k,
                               v in pretrained_dict.items() if k in model_dict}
            # overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            # load the new state dict
            self.model.load_state_dict(pretrained_dict)

        self.optimizer.load_state_dict(state['optimizer'])

        model_id = state['iter_idx']
        return model_id

    def save_state(self, save_location, train_instance_name, iter_idx, epoch):
        filename = f'{train_instance_name}.pth'
        save_path = os.path.join(save_location, filename)
        os.makedirs(save_location, exist_ok=True)
        state = {
            'epoch': epoch,
            'iter_idx': iter_idx,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        torch.save(state, save_path)
