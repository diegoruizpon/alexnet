import torch 
import torch.nn as nn



class yolo(nn.Module):
    def __init__(self):
        super(yolo, self).__init__()
        
        self.leakyRELU = nn.LeakyReLU(negative_slope=0.1)
        
        ## Convolutional layers
        
        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=64,
                               kernel_size=7,
                               stride = 2)
        
        self.conv2 = nn.Conv2d(in_channels=64,
                               out_channels=192,
                               kernel_size=(3,3),
                               )
        
        # Conv 3
        
        self.conv3_1 = nn.Conv2d(in_channels=192,
                               out_channels=128,
                               kernel_size=(1,1),
                               )
                                      
        self.conv3_2 = nn.Conv2d(in_channels=128,
                               out_channels=256,
                               kernel_size=(3,3),
                               )
        
        self.conv3_3 = nn.Conv2d(in_channels=256,
                               out_channels=256,
                               kernel_size=(1,1),
                               )
        
        self.conv3_4 = nn.Conv2d(in_channels=256,
                               out_channels=512,
                               kernel_size=(3,3),
                               )
        
        
        # Conv 4
        
        ## Conv 4.1 x4
        self.conv4_1 = nn.Conv2d(in_channels=512,
                               out_channels=256,
                               kernel_size=(1,1),
                               )
                                      
        self.conv4_2 = nn.Conv2d(in_channels=256,
                               out_channels=512,
                               kernel_size=(3,3),
                               )
        
        ## Conv 4.2
        
        self.conv4_3 = nn.Conv2d(in_channels=512,
                               out_channels=512,
                               kernel_size=(1,1),
                               )
        
        
        self.conv4_4 = nn.Conv2d(in_channels=512,
                               out_channels=1024,
                               kernel_size=(3,3)
                               )
        
        
        # Conv 5
        
        # Conv 5.1 x2
        
        self.conv5_1 = nn.Conv2d(in_channels=1024,
                               out_channels=512,
                               kernel_size=(1,1),
                               )
        
        self.conv5_2 = nn.Conv2d(in_channels=512,
                               out_channels=1024,
                               kernel_size=(3,3),
                               )
        
        # conv 5.2
        
        
        self.conv5_3 = nn.Conv2d(in_channels=1024,
                               out_channels=1024,
                               kernel_size=(3,3),
                               )
        
        self.conv5_4 = nn.Conv2d(in_channels=1024,
                               out_channels=1024,
                               kernel_size=3,
                               stride = 2
                               )
        
        # Conv 6 
        
        self.conv6_1 = nn.Conv2d(in_channels=1024,
                               out_channels=1024,
                               kernel_size=(3,3),
                               )
        self.conv6_2 = nn.Conv2d(in_channels=1024,
                               out_channels=1024,
                               kernel_size=(3,3),
                               )
        
        
        ## Linear layers
        
        self.linear1 = nn.Linear(in_features=1024, out_features=4096)
        
        
        self.linear2 = nn.Linear(in_features=4096, out_features=30)
        
        self.maxpool = nn.MaxPool2d(kernel_size=2,
                                    stride = 2)
        
        
        
    def forward(self, x): 
        # Conv 1
        x = self.maxpool(self.leakyRELU(self.conv1(x)))
        
        #Conv 2
        x = self.maxpool(self.leakyRELU(self.conv2(x)))
        
        # Conv 3
        x = self.leakyRELU(self.conv3_1(x))
        x = self.leakyRELU(self.conv3_2(x))
        x = self.leakyRELU(self.conv3_3(x))
        x = self.leakyRELU(self.conv3_4(x))
        x = self.maxpool(x)
        
        # Conv 4
        for _ in range(4):
            x = self.leakyRELU(self.conv4_1(x))
            x = self.leakyRELU(self.conv4_2(x))
            
        x = self.leakyRELU(self.conv4_3(x))
        x = self.leakyRELU(self.conv4_4(x))
        x = self.maxpool(x)
        
        # Conv 5
        for _ in range(2):
            x = self.leakyRELU(self.conv5_1(x))
            x = self.leakyRELU(self.conv5_2(x))
            
        x = self.leakyRELU(self.conv5_3(x))
        # x = self.leakyRELU(self.conv5_4(x))
        
        # # Conv 6
        
        # x = self.leakyRELU(self.conv6_1(x))
        # x = self.leakyRELU(self.conv6_2(x))
        
        return x
        
        


sample = torch.randn(3, 454, 454)


model = yolo()

print(model(sample).shape)
        
        
    
        
        
        
        
        
