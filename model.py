import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        self.drop_prob = 0.1
        
        #  define the LSTM
        self.lstm = nn.LSTM(self.embed_size, self.hidden_size, self.num_layers, 
                            self.drop_prob, batch_first=True)
        self.dropout = nn.Dropout(self.drop_prob)
        self.fc = nn.Linear(hidden_size, self.vocab_size)
        self.embed = nn.Embedding(self.vocab_size, self.embed_size)
        
        # initialize the weights
        self.init_weights()
        
    def init_weights(self):
        # Set bias tensor to all zeros
        self.fc.bias.data.fill_(0)
        # FC weights as random uniform
        self.fc.weight.data.uniform_(-1, 1)
        # FC weights as random uniform
        self.embed.weight.data.uniform_(-1, 1)
    
    def forward(self, features, captions):
        # Remove end-token from all captions
        embeds = self.embed(captions)
        
        features = features.unsqueeze(1)
        
        #concat features and embedded caption
        merged = torch.cat((features, embeds[:, :-1,:]), dim=1)
        
        hiddens, states = self.lstm(merged)
        
        outputs = self.fc(hiddens)
        
        return outputs
        

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        #empty list
        outputs_list = []
        hidden = (torch.randn(self.num_layers, 1, self.hidden_size).to(inputs.device),
                  torch.randn(self.num_layers, 1, self.hidden_size).to(inputs.device))

        for i in range(max_len):
            # LSTM
            # Add the extra 2nd dimension
            out, hidden = self.lstm(inputs, hidden) 
            
            # linear layer with squeezed input
            outputs = self.fc(out.squeeze(1)) 
            
            # get th emax
            pred = outputs.argmax(dim=1)    
            outputs_list.append(pred.item())
            
            inputs = self.embed(pred.unsqueeze(0))
       
        outputs_list = [int(i) for i in outputs_list]
        return  outputs_list
